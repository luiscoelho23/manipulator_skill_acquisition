import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from ament_index_python.packages import get_package_share_directory
from manipulator import kdl_parser

def print_usage():
    print("Usage: python trajectory_mapping.py <ee_trajectory.csv> <human_angles.xlsx> <output_file.csv>")
    print("Example: python trajectory_mapping.py ee_trajectory.csv human_angles.xlsx robot_trajectory.csv")
    print("\nArguments:")
    print("  ee_trajectory.csv: CSV file containing end-effector trajectory (X,Z coordinates)")
    print("  human_angles.xlsx: Excel file containing human joint angles")
    print("  output_file.csv: Output CSV file for robot joint angles")
    print("\nThe script will also create a second output file with '_ee.csv' suffix")
    print("containing the actual end-effector trajectory.")
    sys.exit(1)

# Verificar argumentos
if len(sys.argv) != 4:
    print("Error: Wrong number of arguments")
    print_usage()

ee_traj_file = sys.argv[1]
human_angles_file = sys.argv[2]
output_file = sys.argv[3]

# Verificar se os arquivos existem e têm os formatos corretos
if not os.path.exists(ee_traj_file):
    print(f"Error: File {ee_traj_file} does not exist")
    print_usage()
if not os.path.exists(human_angles_file):
    print(f"Error: File {human_angles_file} does not exist")
    print_usage()
if not ee_traj_file.endswith('.csv'):
    print(f"Error: {ee_traj_file} must be a CSV file")
    print_usage()
if not human_angles_file.endswith('.xlsx'):
    print(f"Error: {human_angles_file} must be an Excel file (.xlsx)")
    print_usage()

try:
    # Função de cinemática direta (para calcular a posição cartesiana a partir dos ângulos das juntas)
    def forward_kinematics(angles):
        joint_angles = kdl.JntArray(7)
        joint_angles[0] = (-180) * np.pi /180  # Joint 1 angle in radians
        joint_angles[1] = (angles[0]) * np.pi /180  # Joint 2 angle in radians
        joint_angles[2] = (0) * np.pi /180  # Joint 3 angle in radians
        joint_angles[3] = (-180 + angles[1]) * np.pi /180  # Joint 4 angle in radians
        joint_angles[4] = (0) * np.pi /180  # Joint 5 angle in radians
        joint_angles[5] = (135 + angles[2]) * np.pi /180  # Joint 6 angle in radians
        joint_angles[6] = 0 * np.pi /180  # Joint 7 angle in radians
        
        fk_solver = kdl.ChainFkSolverPos_recursive(kdl_chain)
        eeframe = kdl.Frame()
        fk_solver.JntToCart(joint_angles, eeframe)
        
        return np.array([eeframe.p.x(), eeframe.p.y(), eeframe.p.z()])

    # Função de cinemática inversa aproximada usando gradiente descendente
    def gradient_descent_ik(joint_angles, target_position, learning_rate=0.01, tolerance=2e-2):
        max_iterations = 50000  # Limite de iterações para convergência
        joint_angle_og = joint_angles.copy()
        for i in range(max_iterations):
            # Calcular a posição cartesiana atual
            current_position = forward_kinematics(joint_angles)
            
            # Calcular o erro total
            error_magnitude = np.linalg.norm(target_position[[0,-1]] - current_position[[0,-1]]) + np.linalg.norm(joint_angle_og - joint_angles) / (1000 * 2000)

            # Parar se o erro for menor que a tolerância
            if error_magnitude < tolerance:
                break 

            # Aproximação do gradiente por diferenças finitas
            gradient = np.zeros(3)
            for i in range(3):
                joint_angles_temp = joint_angles.copy()
                joint_angles_temp[i] += 0.1  # Pequena variação para estimar o gradiente
                new_position = forward_kinematics(joint_angles_temp)
                gradient[i] = (np.linalg.norm(target_position[[0,-1]] - new_position[[0,-1]] + np.linalg.norm(joint_angle_og - joint_angles_temp) / (1000 * 2000)) - error_magnitude) / 0.001

            # Atualizar ângulos das juntas na direção oposta ao gradiente para minimizar o erro
            joint_angles -= learning_rate * gradient

            if i > max_iterations - 2:
                print("!!!Couldn't converge!!!")
                time.sleep(10)
        
        return joint_angles

    #Load Robot Kinematics
    package_share_dir = get_package_share_directory('manipulator')
    robot_urdf_path = os.path.join(package_share_dir, 'resources', 'robot_description', 'manipulator.urdf')
    robot = URDF.from_xml_file(robot_urdf_path)
    (_,kdl_tree) = kdl_parser.treeFromUrdfModel(robot)
    kdl_chain = kdl_tree.getChain("panda_link0", "panda_finger")

    # Carregar trajetórias
    try:
        # Trajetória desejada em coordenadas cartesianas
        #trajectory_desired_ee = np.array([[-0.5 + 0.03 * t, 0 , 0.5 * np.sin(t)] for t in range(100)])
        # Trajetória desejada em coordenadas cartesianas
        trajectory_desired_ee = np.genfromtxt(ee_traj_file, delimiter=',', skip_header=1)
        
        # Trajetória desejada em espaço de juntas
        trajectory_desired_ang = pd.read_excel(human_angles_file, header=None)
    except Exception as e:
        print(f"Error reading input files: {str(e)}")
        print_usage()

    # Ângulos iniciais das juntas
    joint_angles = np.array(trajectory_desired_ang)  # Ângulos iniciais
    trajectory_actual_cartesian = []
    trajectory_actual_joint_angles = []

    # Loop para seguir a trajetória desejada, ajustando os ângulos para minimizar o erro
    i = 0
    for desired_position in trajectory_desired_ee:
        # Atualizar os ângulos das juntas usando o algoritmo de gradiente descendente
        joint_angles[i] = gradient_descent_ik(joint_angles[i], desired_position)    

        # Salvar a posição atual em angulos    
        trajectory_actual_joint_angles.append(joint_angles[i])

        # Calcular a posição atual após ajuste dos ângulos
        current_position = forward_kinematics(joint_angles[i])

        # Salvar a posição atual calculada para visualização
        trajectory_actual_cartesian.append(current_position)
        i += 1
        print("Step " + str(i) + " Done")

    # Converte a lista de trajetórias reais para um array numpy
    trajectory_actual_cartesian = np.array(trajectory_actual_cartesian)
    trajectory_actual_joint_angles = np.array(trajectory_actual_joint_angles)

    # Salvar resultados
    try:
        np.savetxt(output_file, trajectory_actual_joint_angles, delimiter=",")
        np.savetxt(output_file.replace(".csv", "_ee.csv"), trajectory_actual_cartesian, delimiter=",")
        print(f"Trajectories saved to {output_file} and {output_file.replace('.csv', '_ee.csv')}")
    except Exception as e:
        print(f"Error saving output files: {str(e)}")
        print_usage()

    # Visualização das trajetórias
    plt.figure(figsize=(12, 6))
    plt.plot(trajectory_desired_ee[:, 0], trajectory_desired_ee[:, 2], label="Reference", linestyle="--")
    plt.plot(trajectory_actual_cartesian[:, 0], trajectory_actual_cartesian[:, 2], label="Robot EE", linestyle="-")
    plt.xlabel("X (metros)")
    plt.ylabel("Z (metros)")
    plt.legend()
    plt.grid()
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
    print_usage()