## Velocity Profile Analysis

### analyze_correlation_velocity_profile.py
This script analyzes the velocity profiles of both end-effector and joint movements, comparing two trajectories (Traj1 and Traj2) for each participant and trial. It generates:

1. End-effector velocity profiles:
   - Shows velocity magnitude over time for both trajectories
   - Calculates mean velocity and velocity consistency
   - Visualizes velocity patterns in mm/s

2. Joint velocity profiles:
   - Displays velocity profiles for all three joints
   - Shows how joint velocities change over time
   - Visualizes velocity patterns in degrees/s

3. Joint correlation analysis:
   - Calculates correlations between joint velocities
   - Shows how joints are coordinated in each trajectory
   - Visualizes correlation patterns using heatmaps

The script processes data for all participants and trials, generating visualizations and statistics that help understand:
- How velocity profiles differ between trajectories
- How joint movements are coordinated
- How these patterns vary across participants and trials

All visualizations are saved in the `visualizations_joint_correlations_velocity_profiles` directory, with detailed statistics logged for further analysis. 