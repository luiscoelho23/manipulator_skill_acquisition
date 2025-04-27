import pandas as pd
import numpy as np
import sys


if len(sys.argv) < 4:
    print("ERROR ARGS")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
output_file = sys.argv[3]


# Read the CSV files without headers and assign column names
traj1 = pd.read_csv(file1)
traj2 = pd.read_csv(file2, header=None, names=['X', 'Y', 'Z'])

# Ensure both trajectories have the same length
min_length = min(len(traj1), len(traj2))
traj1 = traj1[:min_length]
traj2 = traj2[:min_length]

# Calculate the difference between the trajectories (Euclidean distance, using only X and Z)
difference = np.sqrt((traj1['X'] - traj2['X'])**2 + (traj1['Z'] - traj2['Z'])**2)

# Calculate mean error and standard deviation
mean_error = np.mean(difference)
std_error = np.std(difference)

trajectory_df = pd.DataFrame({
    "Sample": range(1, len(traj1) + 1),
    "Traj1_X": traj1['X'],
    "Traj1_Z": traj1['Z'],
    "Traj2_X": traj2['X'],
    "Traj2_Z": traj2['Z'],
})

summary_df = pd.DataFrame({
    "Mean_Error": [mean_error],
    "Std_Error": [std_error],
})

with pd.ExcelWriter(output_file) as writer:
    # Write the summary at the top
    summary_df.to_excel(writer, sheet_name="Results", index=False, startrow=0)
    
    # Write the trajectory data just below the summary (add +2 to leave a blank row)
    trajectory_df.to_excel(writer, sheet_name="Results", index=False, startrow=len(summary_df) + 2)


print("Difference calculated and saved to " + output_file)