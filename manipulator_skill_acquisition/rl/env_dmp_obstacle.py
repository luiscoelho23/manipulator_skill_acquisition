import os
import sys
import numpy as np

import time
import subprocess

import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
import pygame

from ament_index_python.packages import get_package_share_directory

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
    
    def __init__(self, dmp_path=None, enable_rendering=True):     
        self.main_trajectory_ang = np.array([])
        
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_last_position = np.zeros(2, dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)

        self.target_position = np.zeros(2, dtype=np.float32)
        self.target_last_position = np.zeros(2, dtype=np.float32)
        self.target_velocity = np.zeros(2, dtype=np.float32)
        
        # Obstacle configuration
        self.num_static_obstacles = 2  # Number of static obstacles
        self.num_moving_obstacles = 2  # Number of moving obstacles
        self.obstacle_radius = 0.05  # Radius used for collision detection
        self.obstacle_area = {  # Bounds for obstacle placement
            'x_min': -0.70, 'x_max': -0.10,
            'z_min': -0.30, 'z_max': 0.30
        }
        
        # Movement parameters for dynamic obstacles
        self.obstacle_velocity = []  # Velocity vectors for moving obstacles
        self.obstacle_max_speed = 0.003  # Maximum speed for moving obstacles
        self.obstacle_amplitude = 0.15   # Maximum distance to move
        self.obstacle_movement_type = []  # Type of movement pattern (0: linear, 1: circular, 2: sine wave)
        self.obstacle_original_pos = []   # Original positions for oscillating obstacles
        self.obstacle_phase = []          # Phase for oscillating obstacles
        
        # Initialize obstacles
        self.obstacles = self._generate_obstacles()
        self.closest_obstacle = self.obstacles[0].copy()
        self.obstacle_is_moving = np.array([False] * self.num_static_obstacles + [True] * self.num_moving_obstacles)
        
        self.traj_index = 0
        self.repeat_ob = False
        self.done = False   

        # Create observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)

        # Load robot model
        pkg_share = get_package_share_directory('manipulator')
        urdf_path = pkg_share + '/resources/robot_description/manipulator.urdf'
        self.robot = URDF.from_xml_file(urdf_path)
        (_,self.kdl_tree) = kdl_parser.treeFromUrdfModel(self.robot)
        self.kdl_chain = self.kdl_tree.getChain("panda_link0", "panda_finger")
        
        # Create FK solver once to reuse
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        self.joint_angles = kdl.JntArray(7)
        self.eeframe = kdl.Frame()
        
        # Pre-compute joint angle constants
        self.joint_angle_constants = np.zeros(7, dtype=np.float32)
        self.joint_angle_constants[0] = (-180) * np.pi / 180
        self.joint_angle_constants[2] = 0
        self.joint_angle_constants[4] = 0
        self.joint_angle_constants[6] = 0
        
        # Use the provided DMP path or fall back to default
        pkg_share = get_package_share_directory('manipulator_skill_acquisition')
        if dmp_path and os.path.exists(dmp_path):
            print(f"Loading DMP from provided path: {dmp_path}")
            self.library = motion.mpx.load_from(dmp_path)
        else:
            default_dmp_path = pkg_share + '/resources/dmp/dmp.mpx'
            print(f"Loading DMP from default path: {default_dmp_path}")
            self.library = motion.mpx.load_from(default_dmp_path)
            
        self.policy = self.library.policies[0]
        self.phase = motion.LinearPacer(1.0)
        self.phase.value = 0.1
        self.ts = 0.001  
        self.load_dmp()
        self.policy.reset([117.86, 120.226, 31.3175])
        
        # Rendering related variables
        self.render_mode = enable_rendering
        self.fps = 60 if enable_rendering else 0  # Reduced from 60 to 30 for better performance
        self.window_size = 800
        self.scale = 1 
        self.surf = None
        self.window = None
        self.clock = None
        
        # Cache variables for reward calculation
        self.dist_penalty_factor = 100000
        self.velocity_penalty_factor = 1000000
        self.proximity_reward_factor = 10000
        
        # Define display offsets for visualization - centered on trajectory
        self.display_offset_x = 0.8
        self.display_offset_y = 0.6
        self.zoom_factor = 700.0
        
        # Display enhancements
        self.agent_trail = []
        self.max_trail_length = 1000  # Reduced from 1000 to 500 for better performance
        self.show_grid = True
        self.trajectory_color = (100, 100, 255)
        self.show_obstacle_zone = True
        
        # Debug initialization
        self.debug_counter = 0
        self.position_values = {}
             
    def reset(self, seed=None, options=None):
        # Reset state variables
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_last_position = np.zeros(2, dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)

        self.target_position = np.zeros(2, dtype=np.float32)
        self.target_last_position = np.zeros(2, dtype=np.float32)
        self.target_velocity = np.zeros(2, dtype=np.float32)
        
        # Generate new obstacles for this episode
        self.obstacles = self._generate_obstacles()
        self.closest_obstacle = self.obstacles[0].copy()
        self.obstacle_is_moving = np.array([False] * self.num_static_obstacles + [True] * self.num_moving_obstacles)
        
        self.traj_index = 0
        
        # Reset DMP
        self.policy = self.library.policies[0]
        self.phase = motion.LinearPacer(1.0)
        self.phase.value = 0.1
        self.ts = 0.001 
        self.load_dmp()
        self.policy.reset([117.86, 120.226, 31.3175])
        
        self.repeat_ob = False
        self.done = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Create observation with information about closest obstacle
        return np.array([
            self.agent_position[0],
            self.agent_position[1],
            self.target_position[0],
            self.target_position[1],
            self.phase.value,
            self.closest_obstacle[0],
            self.closest_obstacle[1]
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        
        # Update phase and policy
        self.phase.update(self.ts)
        self.policy.update(self.ts, self.phase.value)
        
        # Update positions of moving obstacles
        self._update_obstacle_positions()

        # Process all obstacles at once for efficiency
        if self.obstacles.shape[0] > 0:
            # Calculate distances to all obstacles efficiently
            delta = np.array([self.agent_position - obstacle for obstacle in self.obstacles])
            distances = np.sqrt(np.sum(delta**2, axis=1))
            
            # Find the closest obstacle
            closest_idx = np.argmin(distances)
            self.closest_obstacle = self.obstacles[closest_idx].copy()
            closest_distance = distances[closest_idx]
            
            # Apply progressive penalties based on proximity to the closest obstacle
            if closest_distance < 0.10:  
                
                self.policy.value = [
                    self.policy.value[0] + action[0],
                    self.policy.value[1] + action[1],
                    self.policy.value[2] + action[2]
                ]
                reward -= 1/closest_distance
            
            if closest_distance < 0.08:
                reward -= 10/closest_distance
            if closest_distance < 0.06:
                reward -= 1000/closest_distance
            if closest_distance < 0.04:
                self.repeat_ob = True
                reward -= 100000/closest_distance
            
            # Add smaller penalties for other obstacles
            for i, dist in enumerate(distances):
                if i != closest_idx and dist < 0.15:  # Only consider other obstacles that are somewhat close
                    reward -= 0.5/dist  # Smaller penalty for non-closest obstacles

        """
        try:
            result = subprocess.run(
                ["./run_cd",[(-180) * np.pi / 180,( self.policy.value[0]  ) * np.pi / 180, 0, (-180 + self.policy.value[1]) * np.pi / 180,
                        0, (135 + self.policy.value[2]) * np.pi / 180, 0 ], [-0.15, 0 , -0.3]],  # Command and arguments
                text=True,                     # Capture output as text (Python 3.7+)
                capture_output=True,           # Capture stdout and stderr
                check=True                     # Raise an exception on non-zero exit code
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            # Handle errors from the C++ program
            print("Error executing C++ program:", e.stderr) 
        """

        # Get current position
        self.agent_position = np.array(self.get_ee_position(
            self.policy.value[0], 
            self.policy.value[1], 
            self.policy.value[2]
        ), dtype=np.float32)
        
        # Calculate velocity, but initialize to zeros on first step
        if hasattr(self, 'agent_last_position') and np.all(np.isfinite(self.agent_last_position)):
            self.agent_velocity = self.agent_position - self.agent_last_position
        else:
            self.agent_velocity = np.zeros(2, dtype=np.float32)
            
        self.agent_last_position = self.agent_position.copy()

        # Get target position
        idx = self.traj_index * 3
        if idx + 2 < len(self.main_trajectory_ang):
            self.target_position = np.array(self.get_ee_position(
                self.main_trajectory_ang[idx],
                self.main_trajectory_ang[idx + 1],
                self.main_trajectory_ang[idx + 2]
            ), dtype=np.float32)
        
        # Calculate target velocity, but initialize to zeros on first step
        if hasattr(self, 'target_last_position') and np.all(np.isfinite(self.target_last_position)):
            self.target_velocity = self.target_position - self.target_last_position
        else:
            self.target_velocity = np.zeros(2, dtype=np.float32)
            
        self.target_last_position = self.target_position.copy()

        # Calculate distance to target
        delta = self.agent_position - self.target_position
        distance_main_trajectory = np.sqrt(np.sum(delta**2))
        
        # Add distance penalty to reward
        reward -= (distance_main_trajectory * self.dist_penalty_factor)
        
        # Add velocity matching penalty, with safety check for NaN values
        velocity_diff = np.sqrt(np.sum((self.agent_velocity - self.target_velocity)**2))
        
        if np.isfinite(velocity_diff):
            reward -= velocity_diff * self.velocity_penalty_factor
        
        # Terminal state condition
        if self.phase.value >= 0.992:
            """
            while True:
                try:
                    user_input = input("Evaluate 1-5: ")
                    evaluation = int(user_input)  # Convert input to an integer
                    if 1 <= evaluation <= 5:  # Check if it's in the range
                        print(f"You entered: {evaluation}")
                        break  # Exit the loop if valid
                except ValueError:
                    print("Invalid input! Please enter a number.")
            reward += 10000 * (evaluation - 3)
            """
            if distance_main_trajectory < 0.05:
                reward += 1/(distance_main_trajectory + 0.01)
            if not self.repeat_ob:    
                self.closest_obstacle = self.obstacles[0].copy()
            self.repeat_ob = False
            self.done = True
       
        self.traj_index += 1
        
        # Only render if rendering is enabled
        if self.render_mode:
            self.render()
        
        return self._get_obs(), reward, self.done, {}, {}

    def render(self):
        if not self.render_mode:
            return
            
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
            
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("DMP Obstacle Environment - X/Z Coordinates")
            
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.window_size, self.window_size))
        self.surf.fill((245, 245, 245))  # Slightly off-white for better contrast
        
        # Apply zoom by scaling coordinates
        def world_to_screen(world_pos):
            """Transform world coordinates to screen coordinates"""
            screen_x = int((world_pos[0] + self.display_offset_x) * self.zoom_factor)
            # Z coordinate goes up in world space, but down in screen space, so we negate it
            screen_y = int((-world_pos[1] + self.display_offset_y) * self.zoom_factor)
            return (screen_x, screen_y)
        
        # Update agent trail
        if len(self.agent_trail) >= self.max_trail_length:
            self.agent_trail.pop(0)  # Remove oldest point
        
        # Add current position to trail in world coordinates (we'll convert when drawing)
        self.agent_trail.append((self.agent_position[0], self.agent_position[1]))
        
        # Draw grid if enabled
        if self.show_grid:
            self.draw_grid()
        
        # Draw coordinate system (axes)
        self.draw_coordinate_system()
        
        # Draw obstacle zones (different danger zones)
        if self.show_obstacle_zone:
            for obstacle in self.obstacles:
                obstacle_pos = world_to_screen((obstacle[0], obstacle[1]))
                
                # Extreme danger zone (< 0.04)
                danger_extreme = pygame.Surface((200, 200), pygame.SRCALPHA)
                pygame.draw.circle(danger_extreme, (255, 0, 0, 40), (100, 100), 
                                 int(0.04 * self.zoom_factor))
                self.surf.blit(danger_extreme, 
                             (obstacle_pos[0] - 100, obstacle_pos[1] - 100))
                
                # High danger zone (< 0.06)
                danger_high = pygame.Surface((240, 240), pygame.SRCALPHA)
                pygame.draw.circle(danger_high, (255, 50, 0, 25), (120, 120), 
                                 int(0.06 * self.zoom_factor))
                self.surf.blit(danger_high, 
                             (obstacle_pos[0] - 120, obstacle_pos[1] - 120))
                
                # Medium danger zone (< 0.08)
                danger_med = pygame.Surface((280, 280), pygame.SRCALPHA)
                pygame.draw.circle(danger_med, (255, 100, 0, 15), (140, 140), 
                                 int(0.08 * self.zoom_factor))
                self.surf.blit(danger_med, 
                             (obstacle_pos[0] - 140, obstacle_pos[1] - 140))
                
                # Low danger zone (< 0.10)
                danger_low = pygame.Surface((320, 320), pygame.SRCALPHA)
                pygame.draw.circle(danger_low, (255, 150, 0, 10), (160, 160), 
                                 int(0.10 * self.zoom_factor))
                self.surf.blit(danger_low, 
                             (obstacle_pos[0] - 160, obstacle_pos[1] - 160))
                
                # Mark closest obstacle with a different color
                if np.array_equal(obstacle, self.closest_obstacle):
                    outline = pygame.Surface((50, 50), pygame.SRCALPHA)
                    pygame.draw.circle(outline, (255, 255, 0, 150), (25, 25), 22)
                    self.surf.blit(outline, (obstacle_pos[0] - 25, obstacle_pos[1] - 25))
        
        # Draw agent trail (trajectory)
        if len(self.agent_trail) > 1:
            for i in range(1, len(self.agent_trail)):
                alpha = int(255 * (i / len(self.agent_trail)))  # Fade older points
                color = (self.trajectory_color[0], self.trajectory_color[1], 
                         min(255, self.trajectory_color[2] + alpha // 3))
                
                # Convert world to screen coordinates
                start_pos = world_to_screen(self.agent_trail[i-1])
                end_pos = world_to_screen(self.agent_trail[i])
                
                pygame.draw.line(self.surf, color, start_pos, end_pos, 2)
        
        # Draw all obstacles
        for i, obstacle in enumerate(self.obstacles):
            obstacle_pos = world_to_screen((obstacle[0], obstacle[1]))
            is_moving = self.obstacle_is_moving[i]
            
            # Draw multiple circles with decreasing opacity for glow effect
            for radius in range(25, 15, -3):
                alpha = 100 - (radius - 15) * 10
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                
                if np.array_equal(obstacle, self.closest_obstacle):
                    # Closest obstacle with a different color
                    if is_moving:
                        pygame.draw.circle(s, (255, 100, 0, alpha), (radius, radius), radius)
                    else:
                        pygame.draw.circle(s, (200, 100, 0, alpha), (radius, radius), radius)
                else:
                    if is_moving:
                        pygame.draw.circle(s, (0, 150, 250, alpha), (radius, radius), radius)  # Blue for moving
                    else:
                        pygame.draw.circle(s, (250, 0, 0, alpha), (radius, radius), radius)    # Red for static
                        
                self.surf.blit(s, (obstacle_pos[0]-radius, obstacle_pos[1]-radius))
            
            # Main obstacle
            if np.array_equal(obstacle, self.closest_obstacle):
                # Closest obstacle - orange color regardless of type
                pygame.draw.circle(self.surf, (255, 120, 0), obstacle_pos, 17)
                pygame.draw.circle(self.surf, (230, 80, 0), obstacle_pos, 15)
                
                # Label the closest obstacle
                obs_font = pygame.font.Font(None, 20)
                obs_label = obs_font.render("CLOSEST", True, (150, 50, 0))
                self.surf.blit(obs_label, (obstacle_pos[0] - 35, obstacle_pos[1] - 30))
            else:
                if is_moving:
                    # Moving obstacle - blue color
                    pygame.draw.circle(self.surf, (30, 100, 255), obstacle_pos, 15)  # Outer edge
                    pygame.draw.circle(self.surf, (0, 50, 200), obstacle_pos, 13)    # Inner fill
                    
                    # Add movement indicator
                    movement_type = self.obstacle_movement_type[i - self.num_static_obstacles + self.num_static_obstacles] if i >= self.num_static_obstacles else None
                    if movement_type == 0:
                        # Show arrow for linear motion
                        velocity = self.obstacle_velocity[i - self.num_static_obstacles + self.num_static_obstacles]
                        if np.linalg.norm(velocity) > 0:
                            # Normalize and scale velocity for arrow
                            direction = velocity / np.linalg.norm(velocity) * 20
                            end_pos = (int(obstacle_pos[0] + direction[0]), int(obstacle_pos[1] - direction[1]))
                            pygame.draw.line(self.surf, (255, 255, 255), obstacle_pos, end_pos, 2)
                            # Arrow head
                            pygame.draw.polygon(self.surf, (255, 255, 255), [
                                end_pos,
                                (end_pos[0] - 5, end_pos[1] - 5),
                                (end_pos[0] - 5, end_pos[1] + 5)
                            ])
                    elif movement_type == 1:
                        # Show rotating dots for circular motion
                        for dot in range(4):
                            angle = self.obstacle_phase[i - self.num_static_obstacles + self.num_static_obstacles] + dot * np.pi/2
                            dot_x = obstacle_pos[0] + 20 * np.cos(angle)
                            dot_y = obstacle_pos[1] - 20 * np.sin(angle)
                            pygame.draw.circle(self.surf, (255, 255, 255), (int(dot_x), int(dot_y)), 2)
                    elif movement_type == 2:
                        # Show wave pattern for sine motion
                        wave_width = 30
                        for x_offset in range(-wave_width, wave_width + 1, 3):
                            scale = 8
                            y_offset = int(scale * np.sin(self.obstacle_phase[i - self.num_static_obstacles + self.num_static_obstacles] + x_offset/10))
                            pygame.draw.circle(self.surf, (255, 255, 255), (obstacle_pos[0] + x_offset, obstacle_pos[1] + y_offset), 1)
                else:
                    # Static obstacle - red color
                    pygame.draw.circle(self.surf, (250, 50, 50), obstacle_pos, 15)
                    pygame.draw.circle(self.surf, (200, 0, 0), obstacle_pos, 13)
                
                # Number the obstacles
                obs_font = pygame.font.Font(None, 18)
                num_label = obs_font.render(f"{i+1}", True, (255, 255, 255))
                text_w = num_label.get_width()
                text_h = num_label.get_height()
                self.surf.blit(num_label, (obstacle_pos[0] - text_w//2, obstacle_pos[1] - text_h//2))
        
        # Draw target with a pulsating effect
        target_pos = world_to_screen((self.target_position[0], self.target_position[1]))
        
        # Pulsating effect for target
        pulse = (np.sin(time.time() * 5) + 1) / 2  # Oscillate between 0 and 1
        pulse_size = 10 + int(pulse * 5)  # Size varies between 10 and 15
        
        # Target glow
        for radius in range(pulse_size+8, pulse_size-1, -2):
            alpha = 150 - (radius - pulse_size) * 15
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 250, 0, alpha), (radius, radius), radius)
            self.surf.blit(s, (target_pos[0]-radius, target_pos[1]-radius))
            
        # Main target
        pygame.draw.circle(self.surf, (50, 200, 50), target_pos, pulse_size)
        pygame.draw.circle(self.surf, (0, 150, 0), target_pos, pulse_size-2)
        
        # Draw agent with a cool blue effect
        agent_pos = world_to_screen((self.agent_position[0], self.agent_position[1]))
        
        # Agent glow
        for radius in range(15, 8, -2):
            alpha = 150 - (radius - 8) * 15
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 100, 250, alpha), (radius, radius), radius)
            self.surf.blit(s, (agent_pos[0]-radius, agent_pos[1]-radius))
            
        # Main agent
        pygame.draw.circle(self.surf, (30, 100, 255), agent_pos, 10)
        pygame.draw.circle(self.surf, (0, 50, 200), agent_pos, 8)
        
        # Calculate distance between agent and target
        distance = np.sqrt(np.sum((self.agent_position - self.target_position)**2))
        
        # Draw a line between agent and target
        pygame.draw.line(self.surf, (100, 100, 100, 150), agent_pos, target_pos, 1)
        
        # Calculate distance to obstacle
        obstacle_distance = np.sqrt(np.sum((self.agent_position - self.closest_obstacle)**2))
        
        # Calculate danger level based on obstacle distance
        danger_level = "None"
        danger_color = (255, 255, 255)
        if obstacle_distance < 0.04:
            danger_level = "EXTREME!"
            danger_color = (255, 0, 0)
        elif obstacle_distance < 0.06:
            danger_level = "High"
            danger_color = (255, 100, 0)
        elif obstacle_distance < 0.08:
            danger_level = "Medium"
            danger_color = (255, 150, 0)
        elif obstacle_distance < 0.10:
            danger_level = "Low"
            danger_color = (255, 200, 0)
        
        # No need to flip the display anymore since we're handling coordinates correctly
        # Display info panel directly
        self.draw_info_panel(distance, obstacle_distance, danger_level, danger_color)
        
        # Blit to screen
        self.window.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.fps)
        pygame.display.flip()
    
    def draw_coordinate_system(self):
        """Draw coordinate system to help visualize the X-Z space"""
        # Function to convert world to screen coordinates 
        def world_to_screen(world_pos):
            screen_x = int((world_pos[0] + self.display_offset_x) * self.zoom_factor)
            screen_y = int((-world_pos[1] + self.display_offset_y) * self.zoom_factor)
            return (screen_x, screen_y)
            
        # Draw origin
        origin_pos = world_to_screen((0, 0))
        pygame.draw.circle(self.surf, (0, 0, 0), origin_pos, 6)
        pygame.draw.circle(self.surf, (200, 200, 200), origin_pos, 4)
        
        # X-axis (red)
        x_end_pos = world_to_screen((0.2, 0))
        pygame.draw.line(self.surf, (200, 0, 0), origin_pos, x_end_pos, 3)
        
        # Z-axis (green) - pointing up in world space
        z_end_pos = world_to_screen((0, 0.2))
        pygame.draw.line(self.surf, (0, 200, 0), origin_pos, z_end_pos, 3)
        
        # Add arrow tips to axes
        # X-axis arrow
        arrow_size = 10
        pygame.draw.polygon(self.surf, (200, 0, 0), [
            x_end_pos,
            (x_end_pos[0] - arrow_size, x_end_pos[1] - arrow_size//2),
            (x_end_pos[0] - arrow_size, x_end_pos[1] + arrow_size//2)
        ])
        
        # Z-axis arrow
        pygame.draw.polygon(self.surf, (0, 200, 0), [
            z_end_pos,
            (z_end_pos[0] - arrow_size//2, z_end_pos[1] + arrow_size),
            (z_end_pos[0] + arrow_size//2, z_end_pos[1] + arrow_size)
        ])
        
        # Add labels
        font = pygame.font.Font(None, 24)
        x_label = font.render("X", True, (200, 0, 0))
        self.surf.blit(x_label, (x_end_pos[0] + 5, x_end_pos[1] - 10))
        
        z_label = font.render("Z", True, (0, 200, 0))
        self.surf.blit(z_label, (z_end_pos[0] - 10, z_end_pos[1] - 25))
    
    def draw_grid(self):
        """Draw a grid to help with spatial awareness"""
        # Function to convert world to screen coordinates 
        def world_to_screen(world_pos):
            screen_x = int((world_pos[0] + self.display_offset_x) * self.zoom_factor)
            screen_y = int((-world_pos[1] + self.display_offset_y) * self.zoom_factor)
            return (screen_x, screen_y)
        
        # Calculate grid spacing (0.1 units in world space)
        grid_spacing = 0.1
        
        # Draw grid lines (faded)
        grid_color = (220, 220, 220)
        
        # Main axes
        origin_pos = world_to_screen((0, 0))
        
        # Vertical lines (X axis)
        for i in range(-8, 9):
            world_x = i * grid_spacing
            start_pos = world_to_screen((world_x, -1.0))
            end_pos = world_to_screen((world_x, 1.0))
            
            # Make origin line darker
            line_color = (180, 180, 180) if i == 0 else grid_color
            line_width = 2 if i == 0 else 1
            pygame.draw.line(self.surf, line_color, start_pos, end_pos, line_width)
            
            # Add coordinate label for major grid lines
            if i % 2 == 0 and i != 0:
                label = pygame.font.Font(None, 18).render(f"{world_x:.1f}", True, (100, 100, 100))
                label_pos = (start_pos[0] - 10, origin_pos[1] + 10)
                self.surf.blit(label, label_pos)
        
        # Horizontal lines (Z axis)
        for i in range(-8, 9):
            world_z = i * grid_spacing
            start_pos = world_to_screen((-1.0, world_z))
            end_pos = world_to_screen((1.0, world_z))
            
            # Make origin line darker
            line_color = (180, 180, 180) if i == 0 else grid_color
            line_width = 2 if i == 0 else 1
            pygame.draw.line(self.surf, line_color, start_pos, end_pos, line_width)
            
            # Add coordinate label for major grid lines
            if i % 2 == 0 and i != 0:
                label = pygame.font.Font(None, 18).render(f"{world_z:.1f}", True, (100, 100, 100))
                label_pos = (origin_pos[0] + 10, start_pos[1] - 10)
                self.surf.blit(label, label_pos)
    
    def draw_info_panel(self, distance, obstacle_distance, danger_level, danger_color):
        """Draw an information panel with useful metrics"""
        # Create panel background
        panel_rect = pygame.Rect(10, 10, 270, 210)  # Increased width from 250 to 300 and height from 200 to 220
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill((30, 30, 30, 180))
        self.surf.blit(s, panel_rect)
        
        # Add text information
        font = pygame.font.Font(None, 22)
        y_offset = 20
        line_spacing = 25
        
        # Agent position
        agent_text = font.render(f"Agent: X={self.agent_position[0]:.2f}, Z={self.agent_position[1]:.2f}", 
                               True, (255, 255, 255))
        self.surf.blit(agent_text, (20, y_offset))
        y_offset += line_spacing
        
        # Target position
        target_text = font.render(f"Target: X={self.target_position[0]:.2f}, Z={self.target_position[1]:.2f}", 
                                True, (255, 255, 255))
        self.surf.blit(target_text, (20, y_offset))
        y_offset += line_spacing
        
        # Distance information
        distance_text = font.render(f"Distance: {distance:.3f} | Obstacle: {obstacle_distance:.3f}", 
                                  True, (255, 255, 255))
        self.surf.blit(distance_text, (20, y_offset))
        y_offset += line_spacing
        
        # Danger level
        danger_text = font.render(f"Danger Level: {danger_level}", True, danger_color)
        self.surf.blit(danger_text, (20, y_offset))
        y_offset += line_spacing
        
        # Obstacle information
        static_count = np.sum(~self.obstacle_is_moving)
        moving_count = np.sum(self.obstacle_is_moving)
        obstacles_text = font.render(f"Obstacles: {len(self.obstacles)} ({static_count} static, {moving_count} moving)", 
                                   True, (255, 255, 255))
        self.surf.blit(obstacles_text, (20, y_offset))
        y_offset += line_spacing
        
        # Info about closest obstacle
        closest_idx = np.where(np.all(self.obstacles == self.closest_obstacle, axis=1))[0][0]
        closest_type = "Moving" if self.obstacle_is_moving[closest_idx] else "Static"
        closest_text = font.render(f"Closest: #{closest_idx+1} ({closest_type})", True, (255, 255, 255))
        self.surf.blit(closest_text, (20, y_offset))
        y_offset += line_spacing
        
        # Joint angles
        joint_text = font.render(f"Joint angles: {self.policy.value[0]:.1f}, {self.policy.value[1]:.1f}, {self.policy.value[2]:.1f}", 
                                True, (255, 255, 255))
        self.surf.blit(joint_text, (20, y_offset))
        y_offset += line_spacing
        
        # Phase percentage
        completion = self.phase.value * 100
        progress_text = font.render(f"Progress: {completion:.1f}%", True, (255, 255, 255))
        self.surf.blit(progress_text, (20, y_offset))
        
        # Phase information in a separate area
        phase_rect = pygame.Rect(10, self.window_size - 40, 200, 30)  # Increased width from 150 to 200
        s = pygame.Surface((phase_rect.width, phase_rect.height), pygame.SRCALPHA)
        s.fill((30, 30, 30, 180))
        self.surf.blit(s, phase_rect)
        
        # Calculate percentage completion
        completion = self.phase.value * 100
        phase_text = font.render(f"Phase: {completion:.1f}%", True, (255, 255, 255))
        self.surf.blit(phase_text, (20, self.window_size - 32))
        
        # Add obstacle avoidance indicator in the corner
        if obstacle_distance < 0.10:
            # Warning indicator
            warning_rect = pygame.Rect(self.window_size - 180, 10, 170, 30)  # Increased width from 150 to 170
            s = pygame.Surface((warning_rect.width, warning_rect.height), pygame.SRCALPHA)
            s.fill((danger_color[0], danger_color[1], danger_color[2], 180))
            self.surf.blit(s, warning_rect)
            
            warning_text = font.render("OBSTACLE NEARBY!", True, (0, 0, 0))
            self.surf.blit(warning_text, (self.window_size - 170, 15))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def load_dmp(self):
        policy = self.library.policies[0]
        
        self.policy.reset([117.86, 120.226, 31.3175])
        
        phase = motion.LinearPacer(1.0)
        phase.value = 0.1
        
        # Pre-allocate trajectory array with estimated size
        estimated_steps = int(0.992 / self.ts)
        trajectory = np.zeros(estimated_steps * 3, dtype=np.float32)
        
        step_count = 0
        while phase.value < 0.992:
            phase.update(self.ts)
            policy.update(self.ts, phase.value)
            
            idx = step_count * 3
            if idx + 2 < len(trajectory):
                trajectory[idx] = policy.value[0]
                trajectory[idx + 1] = policy.value[1]
                trajectory[idx + 2] = policy.value[2]
                step_count += 1
            else:
                # Extend array if needed
                extension = np.zeros(estimated_steps * 3, dtype=np.float32)
                trajectory = np.concatenate([trajectory, extension])
                continue
        
        # Trim to actual size
        self.main_trajectory_ang = trajectory[:step_count * 3]
        
        # Initialize positions
        self.agent_position = np.array(self.get_ee_position(
            self.main_trajectory_ang[0],
            self.main_trajectory_ang[1],
            self.main_trajectory_ang[2]
        ), dtype=np.float32)
        
        self.target_position = self.agent_position.copy()
        
        # Reset trail when loading new DMP
        self.agent_trail = []
    

    def get_ee_position(self, ang1, ang2, ang3):      
        # Re-use pre-allocated joint angles array
        self.joint_angles[0] = self.joint_angle_constants[0]
        self.joint_angles[1] = ang1 * np.pi / 180
        self.joint_angles[2] = self.joint_angle_constants[2]
        self.joint_angles[3] = (-180 + ang2) * np.pi / 180
        self.joint_angles[4] = self.joint_angle_constants[4]
        self.joint_angles[5] = (135 + ang3) * np.pi / 180
        self.joint_angles[6] = self.joint_angle_constants[6]
        
        # Compute forward kinematics
        self.fk_solver.JntToCart(self.joint_angles, self.eeframe)
        
        return self.eeframe.p.x(), self.eeframe.p.z()

    def _generate_obstacles(self):
        """Generate random obstacles within the defined area"""
        obstacles = []
        self.obstacle_velocity = []
        self.obstacle_movement_type = []
        self.obstacle_original_pos = []
        self.obstacle_phase = []
        
        # Generate static obstacles
        for _ in range(self.num_static_obstacles):
            x = np.random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'])
            z = np.random.uniform(self.obstacle_area['z_min'], self.obstacle_area['z_max'])
            obstacles.append(np.array([x, z], dtype=np.float32))
            # Add zero velocity for static obstacles
            self.obstacle_velocity.append(np.zeros(2, dtype=np.float32))
            self.obstacle_movement_type.append(None)
            self.obstacle_original_pos.append(None)
            self.obstacle_phase.append(None)
        
        # Generate moving obstacles
        for _ in range(self.num_moving_obstacles):
            x = np.random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'])
            z = np.random.uniform(self.obstacle_area['z_min'], self.obstacle_area['z_max'])
            obstacles.append(np.array([x, z], dtype=np.float32))
            
            # Randomize movement pattern
            movement_type = np.random.randint(0, 3)  # 0: linear, 1: circular, 2: sine wave
            self.obstacle_movement_type.append(movement_type)
            self.obstacle_original_pos.append(np.array([x, z], dtype=np.float32))
            
            # Initialize velocity based on movement type
            if movement_type == 0:  # Linear
                # Random direction
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0.001, self.obstacle_max_speed)
                vx = speed * np.cos(angle)
                vz = speed * np.sin(angle)
                self.obstacle_velocity.append(np.array([vx, vz], dtype=np.float32))
            else:  # Circular or sine wave
                self.obstacle_velocity.append(np.zeros(2, dtype=np.float32))  # Will be calculated during step
            
            # Random starting phase for movement patterns
            self.obstacle_phase.append(np.random.uniform(0, 2 * np.pi))
            
        return np.array(obstacles)

    def _update_obstacle_positions(self):
        """Update the positions of moving obstacles"""
        for i in range(self.num_static_obstacles, self.num_static_obstacles + self.num_moving_obstacles):
            # Skip if this is not a moving obstacle
            if not self.obstacle_is_moving[i]:
                continue
            
            # Get the movement type
            movement_type = self.obstacle_movement_type[i - self.num_static_obstacles + self.num_static_obstacles]
            original_pos = self.obstacle_original_pos[i - self.num_static_obstacles + self.num_static_obstacles]
            phase = self.obstacle_phase[i - self.num_static_obstacles + self.num_static_obstacles]
            
            # Update phase
            self.obstacle_phase[i - self.num_static_obstacles + self.num_static_obstacles] += 0.01
            phase = self.obstacle_phase[i - self.num_static_obstacles + self.num_static_obstacles]
            
            if movement_type == 0:  # Linear
                # Get velocity
                velocity = self.obstacle_velocity[i - self.num_static_obstacles + self.num_static_obstacles]
                
                # Update position
                self.obstacles[i] += velocity
                
                # Bounce if hitting boundaries
                if (self.obstacles[i][0] <= self.obstacle_area['x_min'] or 
                    self.obstacles[i][0] >= self.obstacle_area['x_max']):
                    velocity[0] *= -1
                if (self.obstacles[i][1] <= self.obstacle_area['z_min'] or 
                    self.obstacles[i][1] >= self.obstacle_area['z_max']):
                    velocity[1] *= -1
                    
            elif movement_type == 1:  # Circular
                # Circular motion around original position
                radius = self.obstacle_amplitude / 2.0
                self.obstacles[i][0] = original_pos[0] + radius * np.cos(phase)
                self.obstacles[i][1] = original_pos[1] + radius * np.sin(phase)
                
            elif movement_type == 2:  # Sine wave
                # Select a random dimension for the sine wave
                if original_pos[0] > self.obstacle_area['x_min'] + 0.2 and original_pos[0] < self.obstacle_area['x_max'] - 0.2:
                    # Horizontal motion (sine wave in x-direction)
                    self.obstacles[i][0] = original_pos[0] + self.obstacle_amplitude/2 * np.sin(phase)
                else:
                    # Vertical motion (sine wave in z-direction)
                    self.obstacles[i][1] = original_pos[1] + self.obstacle_amplitude/2 * np.sin(phase)