import numpy as np
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

def print_usage():
    print("Usage: python fk_human_traj.py <input_file.xlsx> <output_file.csv> <L1> <L2> <L3> <L4>")
    print("Example: python fk_human_traj.py joint_angles.xlsx ee_trajectory.csv 0.3 0.4 0.3 0.2")
    print("\nArguments:")
    print("  input_file.xlsx: Excel file containing joint angles")
    print("  output_file.csv: Output CSV file for end-effector trajectory")
    print("  L1: Back length in meters")
    print("  L2: Shoulder length in meters")
    print("  L3: Arm length in meters")
    print("  L4: Forearm length in meters")
    sys.exit(1)

# Verificar argumentos
if len(sys.argv) != 7:
    print("Error: Wrong number of arguments")
    print_usage()

input_file = sys.argv[1]
output_file = sys.argv[2]

# Verificar se o arquivo de entrada existe e é um arquivo Excel
if not input_file.endswith('.xlsx'):
    print(f"Error: {input_file} must be an Excel file (.xlsx)")
    print_usage()
if not os.path.exists(input_file):
    print(f"Error: File {input_file} does not exist")
    print_usage()

# Verificar se os comprimentos são números válidos
try:
    l1 = float(sys.argv[3])
    l2 = float(sys.argv[4])
    l3 = float(sys.argv[5])
    l4 = float(sys.argv[6])
except ValueError:
    print("Error: Lengths must be valid numbers")
    print_usage()

def Trans(tx, ty, tz):
    M = np.identity(4)
    M[0][3]=tx
    M[1][3]=ty
    M[2][3]=tz
    return M

def Rot(eixo, angulo):
    ang_rad=angulo*math.pi/180.0
    c=math.cos(ang_rad)
    s=math.sin(ang_rad)
    M = np.identity(4)
    if (eixo=='x' or eixo=='X'):
        M[1][1]=M[2][2]=c
        M[1][2]=-s
        M[2][1]=s
    elif (eixo=='y' or eixo=='Y'):
        M[0][0]=M[2][2]=c
        M[0][2]=s
        M[2][0]=-s
    elif (eixo=='z' or eixo=='Z'):
        M[0][0]=M[1][1]=c
        M[0][1]=-s
        M[1][0]=s
    return M

def braco(talfa, tteta, td, tl):
    a=Rot('z', tteta)
    b=Trans(tl, 0, td)
    c=Rot('x', talfa)
    temp = np.dot(a,b)
    T = np.dot(temp,c)
    return T


def robo(DOF, teta, alfa, d, l):
    T=np.identity(4)
    for i in range(DOF):
        A = braco(alfa[i], teta[i], d[i], l[i])
        T = np.dot(T, A)
    return T


def load_jointPositions_dataset():
    try:
        dataset = pd.read_excel(input_file, header=None)
        pelvis_pos = dataset.iloc[:, 0].to_numpy()
        shoulder_pos = dataset.iloc[:, 1].to_numpy()
        elbow_pos = dataset.iloc[:, 2].to_numpy()
        return pelvis_pos, shoulder_pos, elbow_pos
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        print_usage()

def get_ee_position(DOF, pelvis_pos, shoulder_pos, elbow_pos):
    teta = [ 0, 90, pelvis_pos, shoulder_pos, elbow_pos] 
    alfa = [90, 0, -180, 0, 0]
    d = [0, 0, -l2, 0, 0]
    l = [0, 0, l1, -l3, -l4]

    T = robo(DOF, teta, alfa, d, l)
    return np.array([T[0, 3], T[1, 3], T[2, 3]])

try:
    DOF = 5
    pelvis_pos, shoulder_pos, elbow_pos = load_jointPositions_dataset()

    ee_trajectory = np.zeros((len(pelvis_pos), 3))

    for i in range(len(pelvis_pos)):
        ee_trajectory[i] = get_ee_position(DOF, pelvis_pos[i], shoulder_pos[i], elbow_pos[i])

    def save_trajectory_to_csv(ee_trajectory, filename=output_file):
        df = pd.DataFrame(ee_trajectory, columns=['X', 'Y', 'Z'])
        df.to_csv(filename, index=False)
        print(f'Trajectory saved to {filename}')

    save_trajectory_to_csv(ee_trajectory)

    plt.plot(ee_trajectory[:, 0], ee_trajectory[:, 2])
    plt.xlabel('Eixo x')
    plt.ylabel('Eixo z')
    plt.title('Gráfico 2D Simples')
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
    print_usage()

    
    