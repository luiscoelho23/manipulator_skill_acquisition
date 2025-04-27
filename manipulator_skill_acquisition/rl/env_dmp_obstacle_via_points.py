import sys
import numpy as np
import time
from ament_index_python.packages import get_package_share_directory
import gym
from gym.error import DependencyNotInstalled
import os
from gym import spaces

current_file = os.path.abspath(__file__)
if '/install/' in current_file:
    ws_path = current_file[:current_file.find('/install/')]
elif '/src/' in current_file:
    ws_path = current_file[:current_file.find('/src/')]
else:
    print("Error: Could not determine workspace path. Script must be run from install or src directory.")
    sys.exit(1)

mplibrary_path = os.path.join(ws_path, 'build/mplibrary')
sys.path.append(mplibrary_path)
import pymplibrary as motion

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from manipulator import kdl_parser


class DmpObstacleEnv:
    
    def __init__(self, dmp_path=None):     
        self.main_trajectory_ang = np.array([])
        
        self.agent_position = np.empty(2)
        self.agent_last_position = np.empty(2)
        self.agent_velocity = np.empty(2)

        self.target_position = np.empty(2)
        self.target_last_position = np.empty(2)
        self.target_velocity = np.empty(2)
        
        self.obstacles = np.array([(-0.4, -0.15)])
        self.closest_obtacle = np.array([-0.4, -0.15])
        self.traj_index = 0
        self.done = False   
    
        # Create observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(5,))  # Via points needs 5 actions
        
        pkg_share = get_package_share_directory('manipulator')
        urdf_path = pkg_share + '/resources/robot_description/manipulator.urdf'
        self.robot = URDF.from_xml_file(urdf_path)
        (_,self.kdl_tree) = kdl_parser.treeFromUrdfModel(self.robot)
        self.kdl_chain = self.kdl_tree.getChain("panda_link0", "panda_finger")
        
        # Use the provided DMP path or fall back to default
        if dmp_path and os.path.exists(dmp_path):
            print(f"Loading DMP from provided path: {dmp_path}")
            self.library = motion.mpx.load_from(dmp_path)
        else:
            default_dmp_path = '/home/luisc/ws_dmps/src/mplearn/python/dmp.mpx'
            print(f"Loading DMP from default path: {default_dmp_path}")
            self.library = motion.mpx.load_from(default_dmp_path)

        self.policy = self.library.policies[0]
        self.phase = motion.LinearPacer(1.0)
        self.phase.value = 0.1
        self.ts = 0.001 
        self.load_dmp()
        #self.policy.reset([126.86,138.226,27.3175])
        self.policy.reset([95.39636567969238,95.39636567969238,9.219886560012206])
        
        self.policy[0].goal.add(0, 0, 0)
        self.policy[1].goal.add(0, 0, 0)
        self.policy[2].goal.add(0, 0, 0)

        self.render_mode = True
        self.fps = 25000000 # MAX
        self.window_size = 512
        self.scale = 1 
        self.surf = None
        self.window = None
        self.clock = None
             
        
    def reset(self, seed=None, options=None):
        
        self.agent_position = np.empty(2)
        self.agent_last_position = np.empty(2)
        self.agent_velocity = np.empty(2)

        self.target_position = np.empty(2)
        self.target_last_position = np.empty(2)
        self.target_velocity = np.empty(2)
        
        self.traj_index = 0
        
        self.policy = self.library.policies[0]
        self.phase = motion.LinearPacer(1.0)
        self.phase.value = 0.1
        self.ts = 0.001 
        self.load_dmp()
        #self.policy.reset([126.86,138.226,27.3175])
        self.policy.reset([95.39636567969238,95.39636567969238,9.219886560012206])
        
        self.done = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        return self.agent_position[0],self.agent_position[1],self.target_position[0],self.target_position[1], self.phase.value, self.closest_obtacle[0],self.closest_obtacle[1]

    def step(self, action):
        
        reward = 0
        self.phase.update(self.ts)
        self.policy.update(self.ts, self.phase.value)
        
        self.policy[0].goal[0].value = action[0] * 180
        self.policy[1].goal[0].value = action[1] * 180
        self.policy[2].goal[0].value = action[2] * 180
        self.policy[0].goal[0].activation.parameters.width = (action[3] + 2)/4.0
        self.policy[1].goal[0].activation.parameters.width = (action[3] + 2)/4.0
        self.policy[2].goal[0].activation.parameters.width = (action[3] + 2)/4.0
        self.policy[0].goal[0].activation.parameters.center = (action[4] + 2)/4.0
        self.policy[1].goal[0].activation.parameters.center = (action[4] + 2)/4.0
        self.policy[2].goal[0].activation.parameters.center = (action[4] + 2)/4.0

        for obstacle in self.obstacles:
            delta = self.agent_position - obstacle
            dist = np.sqrt(delta[0]**2 + delta[1]**2)
            
            if dist < 0.08:
                reward -= 10/dist
            if dist < 0.06:
                reward -= 1000/dist
            if dist < 0.04:
                reward -= 100000/dist

        # Get current position
        self.agent_position = self.get_ee_position(self.policy.value[0], self.policy.value[1], self.policy.value[2])
        
        # Calculate velocity, but initialize to zeros on first step
        if hasattr(self, 'agent_last_position') and np.all(np.isfinite(self.agent_last_position)):
            self.agent_velocity = np.array(self.agent_position) - np.array(self.agent_last_position)
        else:
            self.agent_velocity = np.zeros(2)
            
        self.agent_last_position = np.array(self.agent_position)

        # Get target position
        self.target_position = self.get_ee_position(
            self.main_trajectory_ang[0 + self.traj_index * 3],
            self.main_trajectory_ang[1 + self.traj_index * 3],
            self.main_trajectory_ang[2 + self.traj_index * 3]
        )
        
        # Calculate target velocity, but initialize to zeros on first step
        if hasattr(self, 'target_last_position') and np.all(np.isfinite(self.target_last_position)):
            self.target_velocity = np.array(self.target_position) - np.array(self.target_last_position)
        else:
            self.target_velocity = np.zeros(2)
            
        self.target_last_position = np.array(self.target_position)

        # Calculate distance to target
        delta = np.array(self.agent_position) - np.array(self.target_position)
        distance_main_trajectory = np.sqrt(delta[0]**2 + delta[1]**2) 
        
        # Add distance penalty to reward
        reward -= (distance_main_trajectory * 100000)
        
        # Add velocity matching penalty, with safety check for NaN values
        velocity_diff = np.sqrt(
            (self.agent_velocity[0] - self.target_velocity[0])**2 + 
            (self.agent_velocity[1] - self.target_velocity[1])**2
        )
        
        if np.isfinite(velocity_diff):
            reward -= velocity_diff * 100000
        
        if self.phase.value >= 0.992:
            if distance_main_trajectory < 0.05:
                reward += 1/(distance_main_trajectory + 0.01)
            reward -= distance_main_trajectory * 1000
            self.done = True
       
        self.traj_index += 1
        
        self.render()
        
        return self._get_obs(), reward, self.done, {},{}

    def render(self):
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
        if self.window is None and self.render_mode:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.window_size, self.window_size))

        pygame.transform.scale(self.surf, (self.scale, self.scale))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())    
        
        pygame.draw.circle(self.surf, (250,0,0), ((self.obstacles[0][0]+0.9) * self.window_size,(self.obstacles[0][1]+0.8) * self.window_size), 17)

        pygame.draw.circle(self.surf, (0,0,255), ((self.agent_position[0]+0.9) * self.window_size,(self.agent_position[1]+0.8) * self.window_size), 5)
        
        pygame.draw.circle(self.surf, (0,255,0), ((self.target_position[0]+0.9) * self.window_size,(self.target_position[1]+0.8) * self.window_size), 5)
        
           
        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode:
            assert self.window is not None
            self.window.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.fps)
            pygame.display.flip()

    def close(self):
        pass

    def load_dmp(self):
        policy = self.library.policies[0]
        
        #policy.reset([126.86,138.226,27.3175])
        self.policy.reset([95.39636567969238,95.39636567969238,9.219886560012206])
        
        phase = motion.LinearPacer(1.0)
        phase.value = 0.1
        
        while phase.value < 0.992:
            phase.update(self.ts)
            policy.update(self.ts, phase.value)
            self.main_trajectory_ang = np.append(self.main_trajectory_ang,[policy.value[0], policy.value[1], policy.value[2]])
        
        self.agent_position = self.get_ee_position(self.main_trajectory_ang[0],self.main_trajectory_ang[1],self.main_trajectory_ang[2])
        self.target_position = self.get_ee_position(self.main_trajectory_ang[0],self.main_trajectory_ang[1],self.main_trajectory_ang[2])

    def get_ee_position(self,ang1,ang2,ang3):      
        
        joint_angles = kdl.JntArray(7)
        joint_angles[0] = (-180) * np.pi /180  # Joint 1 angle in radians
        joint_angles[1] = (ang1) * np.pi /180  # Joint 2 angle in radians
        joint_angles[2] = (0) * np.pi /180  # Joint 3 angle in radians
        joint_angles[3] = (-180 + ang2) * np.pi /180  # Joint 4 angle in radians
        joint_angles[4] = (0) * np.pi /180  # Joint 5 angle in radians
        joint_angles[5] = (135 + ang3) * np.pi /180  # Joint 6 angle in radians
        joint_angles[6] = 0 * np.pi /180  # Joint 7 angle in radians
        
        fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        eeframe = kdl.Frame()
        fk_solver.JntToCart(joint_angles, eeframe)
        
        return eeframe.p.x() , eeframe.p.z()
        
    def load_dmp_from_file(self, file_path):
        """Load DMP from an external file path."""
        print(f"Loading DMP from file: {file_path}")
        try:
            self.library = motion.mpx.load_from(file_path)
            self.policy = self.library.policies[0]
            self.phase = motion.LinearPacer(1.0)
            self.phase.value = 0.1
            self.ts = 0.001
            self.load_dmp()
            self.policy.reset([95.39636567969238,95.39636567969238,9.219886560012206])
            
            # Re-add goal points for via point control
            self.policy[0].goal.add(0, 0, 0)
            self.policy[1].goal.add(0, 0, 0)
            self.policy[2].goal.add(0, 0, 0)
            
            print("DMP loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading DMP from file: {e}")
            return False