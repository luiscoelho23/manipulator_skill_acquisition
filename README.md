# Manipulator Skill Acquisition

A ROS2 package for skill acquisition and learning algorithms for the FRANKA EMIKA Panda robot.

## Overview

This package provides a comprehensive suite of reinforcement learning (RL) models and environments designed for robot skill acquisition. It includes:

- Dynamic Movement Primitives (DMP) integration
- Reinforcement Learning (RL) algorithms:
  - Twin Delayed DDPG (TD3)
  - Deep Deterministic Policy Gradient (DDPG)
  - Soft Actor-Critic (SAC)
- Custom simulation environments for training
- Utilities for model saving and loading
- Data processing tools for motion captured data
- Error evaluation tools for trajectory analysis

## Architecture

The package is organized into the following components:

```
manipulator_skill_acquisition/
├── manipulator_skill_acquisition/    # Python modules
│   ├── rl/                          # Reinforcement learning components
│   │   ├── env_dmp_obstacle.py      # Environment for obstacle avoidance
│   │   ├── env_dmp_obstacle_via_points.py # Via-points environment
│   │   ├── train_td3.py             # TD3 algorithm implementation
│   │   ├── train_ddpg.py            # DDPG algorithm implementation
│   │   ├── train_sac.py             # SAC algorithm implementation
│   │   ├── nn.py                    # Neural network architectures
│   │   └── load_rl.py               # Utilities for loading models
│   ├── SetupMenu/                   # Setup utilities
│   │   ├── excel_avg.py             # Average multiple Excel datasets
│   │   ├── excel_extract.py         # Extract data from Excel files
│   │   ├── dmp_generator.py         # Generate DMPs from trajectories
│   │   ├── trajectory_mapping.py    # Map human to robot trajectories
│   │   └── fk_human_traj.py         # Forward kinematics for human trajectories
│   ├── EvalnRunMenu/                # Evaluation utilities
│   │   ├── error_ee.py              # End-effector error calculation
│   │   └── error_ang.py             # Joint angle error calculation
│   └── __init__.py
├── config_dmp/                      # DMP configuration files
├── config_rl/                       # RL algorithm configuration files
└── resources/                       # Additional resources
```

## Dependencies

### Core Dependencies
- ROS2 Humble
- Python 3.8+
- PyTorch
- NumPy
- pandas (for Excel data processing)
- manipulator package
- mplibrary (for motion primitives)

## Installation

1. Clone the repository into your workspace:
   ```bash
   cd ~/ws_manipulator/src
   git clone https://github.com/luiscoelho23/manipulator_skill_acquisition.git
   ```

2. Install Python dependencies:
   ```bash
   pip install torch numpy gymnasium pandas openpyxl matplotlib
   ```

3. Build the workspace:
   ```bash
   cd ~/ws_manipulator
   colcon build --packages-select manipulator_skill_acquisition
   source install/setup.bash
   ```

## Usage

### Data Processing and DMP Generation

The SetupMenu modules provide tools for processing motion capture data and generating DMPs:

```bash
# Navigate to the SetupMenu directory
cd ~/ws_manipulator/src/manipulator_skill_acquisition/manipulator_skill_acquisition/SetupMenu

# Process Excel data
python3 excel_extract.py input.xlsx output.csv

# Generate DMPs from trajectory data
python3 dmp_generator.py trajectory.csv
```

#### SetupMenu Modules

- **excel_avg.py**: Averages data across multiple Excel files to create a consistent dataset
- **excel_extract.py**: Extracts specific motion data from Xsense Excel exports
- **dmp_generator.py**: Creates Dynamic Movement Primitives from processed trajectory data
- **trajectory_mapping.py**: Maps human demonstration trajectories to robot joint space
- **fk_human_traj.py**: Calculates forward kinematics for human demonstration data

### Trajectory Evaluation

The EvalnRunMenu modules provide tools for evaluating trajectory execution performance:

```bash
# Navigate to the EvalnRunMenu directory
cd ~/ws_manipulator/src/manipulator_skill_acquisition/manipulator_skill_acquisition/EvalnRunMenu

# Calculate end-effector position error
python3 error_ee.py reference.csv actual.csv

# Calculate joint angle error
python3 error_ang.py reference.csv actual.csv
```

#### EvalnRunMenu Modules

- **error_ee.py**: Calculates and visualizes end-effector position errors between reference and actual trajectories
- **error_ang.py**: Calculates and visualizes joint angle errors between reference and actual trajectories

### Training a Reinforcement Learning Model

To train a new RL model for obstacle avoidance:

```bash
# Navigate to the rl directory
cd ~/ws_manipulator/src/manipulator_skill_acquisition/manipulator_skill_acquisition/rl

# Create a configuration file (or use an existing one)
# Config file should be in YAML format with algorithm hyperparameters, 
# DMP settings, and output configuration
nano config_td3.yaml

# Training with TD3 algorithm (provide full path to config file)
python3 train_td3.py /path/to/config_td3.yaml

# Or use DDPG algorithm
python3 train_ddpg.py /path/to/config_ddpg.yaml

# Or use SAC algorithm
python3 train_sac.py /path/to/config_sac.yaml
```

### Configuration

Configuration files can be found in:
- `config_rl/`: Contains parameters for RL algorithms
- `config_dmp/`: Contains DMP configurations

### Loading and Using Models

Models can be loaded in other packages using:

```python
from manipulator_skill_acquisition.rl import load_rl

# Load a trained model
rl_actor = load_rl.load_actor(model_path)

# Get action for current state
action = rl_actor.get_action(state)
```

## Implementation Details

### Skill Acquisition Pipeline

1. **Data Collection**: Collect human demonstration data using Xsense motion capture
2. **Data Processing**: Process and clean the data using excel_extract.py and excel_avg.py
3. **Trajectory Mapping**: Map human trajectories to robot space using trajectory_mapping.py
4. **DMP Generation**: Create DMPs from mapped trajectories using dmp_generator.py
5. **Reinforcement Learning**: Train RL models to adapt DMPs for new scenarios
6. **Evaluation**: Evaluate trajectory execution using error_ee.py and error_ang.py

### Reinforcement Learning Algorithms

The package implements several RL algorithms:

- **TD3**: Twin Delayed DDPG, which addresses overestimation bias in actor-critic methods
- **DDPG**: Deep Deterministic Policy Gradient, an off-policy algorithm for continuous action spaces
- **SAC**: Soft Actor-Critic, which balances exploration and exploitation through entropy maximization

Custom environments are provided that simulate:
- Obstacle avoidance scenarios
- Via-points trajectory following
- Integration with Dynamic Movement Primitives (DMPs)

## License

This project is licensed under the Apache License 2.0.

### Configuration File Format

The training scripts require a YAML configuration file with the following structure:

```yaml
environment: env_dmp_obstacle  # or env_dmp_obstacle_via_points
hyperparameters:
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 256
  tau: 0.01
  buffer_limit: 1000000
  # SAC-specific parameters
  init_alpha: 0.1
  # DDPG/TD3-specific parameters
  exploration_noise: 0.1
  # TD3-specific parameters
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
dmp:
  file: /path/to/dmp_file.mpx
  parameters:
    n_basis: 25
    n_primitives: 3
    phase_bounds: [0.0, 1.0]
    integration_timestep: 0.001
output:
  directory: /path/to/output/dir
  model_name: rl_actor
  save_frequency: 100
  save_checkpoints: true
  save_best: true
logging:
  enabled: true
  directory: logs
  frequency: 10
  tensorboard: true
  console: true
```

You only need to include the algorithm-specific parameters for the algorithm you're using. The GUI provided by `manipulator_gui` can help generate this configuration automatically. 