import pandas as pd
import numpy as np
import sys


if len(sys.argv) < 4:
    print("ERROR ARGS")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
output_file = sys.argv[3]

# Read the Excel file without headers and assign column names
traj1 = pd.read_excel(file1, header=None, names=['Angle1', 'Angle2', 'Angle3'])
traj2 = pd.read_csv(file2, header=None, names=['Angle1', 'Angle2', 'Angle3'])

# Ensure both trajectories have the same length
min_length = min(len(traj1), len(traj2))
traj1 = traj1[:min_length]
traj2 = traj2[:min_length]

# Calculate the difference for each joint angle
diff_angle1 = abs(traj1['Angle1'] - traj2['Angle1'])
diff_angle2 = abs(traj1['Angle2'] - traj2['Angle2'])
diff_angle3 = abs(traj1['Angle3'] - traj2['Angle3'])

# Calculate mean error and standard deviation for each joint angle
mean_error1 = np.mean(diff_angle1)
std_error1 = np.std(diff_angle1)

mean_error2 = np.mean(diff_angle2)
std_error2 = np.std(diff_angle2)

mean_error3 = np.mean(diff_angle3)
std_error3 = np.std(diff_angle3)

trajectory_df = pd.DataFrame({
    "Sample": range(1, len(traj1) + 1),
    "traj1_ang1": traj1["Angle1"],
    "traj1_ang2": traj1["Angle2"],
    "traj1_ang3": traj1["Angle3"],
    "traj2_ang1": traj2["Angle1"],
    "traj2_ang2": traj2["Angle2"],
    "traj2_ang3": traj2["Angle3"],
})

summary_df = pd.DataFrame({
    "Joint": ["Joint1", "Joint2", "Joint3"],
    "Mean_Error": [mean_error1, mean_error2, mean_error3],
    "Std_Error": [std_error1, std_error2, std_error3],
})

with pd.ExcelWriter(output_file) as writer:
    # Write the summary at the top
    summary_df.to_excel(writer, sheet_name="Results", index=False, startrow=0)
    
    # Write the trajectory data just below the summary (add +2 to leave a blank row)
    trajectory_df.to_excel(writer, sheet_name="Results", index=False, startrow=len(summary_df) + 2)

print("Mean and standard deviation for each joint angle saved to" + output_file)
