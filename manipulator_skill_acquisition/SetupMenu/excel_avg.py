import pandas as pd
import numpy as np
import sys
import os

def print_usage():
    print("Usage: python excel_avg.py <input_file1.xlsx> <input_file2.xlsx> ... <output_file.xlsx>")
    print("Example: python excel_avg.py traj1.xlsx traj2.xlsx traj3.xlsx avg_traj.xlsx")
    sys.exit(1)

# Verify arguments
if len(sys.argv) < 3:  # Need at least one input file and one output file
    print("Error: Not enough arguments. Need at least one input file and one output file.")
    print_usage()

# The last argument is the output file
output_file_path = sys.argv[-1]

# All other arguments are input files
file_paths = sys.argv[1:-1]

# Verify if we have at least one input file
if len(file_paths) < 1:
    print("Error: At least one input file is required")
    print_usage()

# Check if the files exist
for file_path in file_paths:
    if not file_path.endswith('.xlsx'):
        print(f"Error: {file_path} must be an Excel file (.xlsx)")
        print_usage()
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        print_usage()

# List to store the DataFrames
dfs = []

# Load the files and store the DataFrames
for file_path in file_paths:
    try:
        # Read Excel file without headers and convert to numeric
        df = pd.read_excel(file_path, header=None)
        # Convert all columns to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Remove any rows that are all NaN
        df = df.dropna(how='all')
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        sys.exit(1)

# Verify if all DataFrames have the same number of columns
if not dfs:
    print("Error: No valid data found in the input files")
    sys.exit(1)

num_cols = dfs[0].shape[1]
for i, df in enumerate(dfs):
    if df.shape[1] != num_cols:
        print(f"Error: File {file_paths[i]} has a different number of columns than the first file")
        sys.exit(1)

# Calculate the average directly without creating an empty DataFrame
df_avg = pd.concat([df for df in dfs], axis=1).groupby(level=0, axis=1).mean()

# Save to a new Excel file without header and without index, starting from the first line
try:
    df_avg.to_excel(output_file_path, index=False, header=False, startrow=0)
    print(f"File saved: {output_file_path}")
    print(f"Successfully averaged {len(file_paths)} files")
except Exception as e:
    print(f"Error saving output file: {str(e)}")
    print_usage()
    sys.exit(1)