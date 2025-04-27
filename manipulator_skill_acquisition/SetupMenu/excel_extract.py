#!/usr/bin/env python3

import sys
import argparse
import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def extract_from_excel(input_file, output_file, columns=None, num_samples=149):
    """
    Extract specific columns from Excel file, interpolate and save to a new file.
    
    Parameters:
    input_file (str): Path to input Excel file
    output_file (str): Path to output Excel file
    columns (list): List of sheet:column pairs to extract (default: None, which uses predefined columns)
    num_samples (int): Number of samples to resample the data to (default: 149)
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return False
    
    try:
        # If no columns specified, use defaults
        if not columns:
            columns = [
                "Joint Angles ZXY:Right Shoulder Flexion/Extension",
                "Joint Angles ZXY:Right Elbow Flexion/Extension",
                "Ergonomic Joint Angles ZXY:Vertical_Pelvis Flexion/Extension"
            ]
        
        # Create a dictionary to store extracted columns
        data = {}
        max_length = 0
        
        # Read each column from the specified sheet
        for col_spec in columns:
            # Parse sheet and column names
            try:
                sheet_name, column_name = col_spec.split(":", 1)
            except ValueError:
                print(f"Error: Invalid column specification '{col_spec}'. Format should be 'SheetName:ColumnName'")
                continue
                
            try:
                # Read the specific sheet
                df = pd.read_excel(input_file, sheet_name=sheet_name)
                
                # Check if the column exists
                if column_name not in df.columns:
                    print(f"Warning: Column '{column_name}' not found in sheet '{sheet_name}'. Skipping.")
                    continue
                
                # Store the column data
                data[column_name] = df[column_name].values
                
                # Update the maximum length
                max_length = max(max_length, len(data[column_name]))
                
            except Exception as e:
                print(f"Error reading sheet '{sheet_name}': {str(e)}")
                continue
        
        if not data:
            print("Error: No valid data was extracted.")
            return False
            
        # Create a dictionary to store interpolated data
        interp_data = {}
        
        # Interpolate each column to have the same length
        for col_name, col_data in data.items():
            # Create x values for the original data
            x_orig = np.linspace(0, 1, len(col_data))
            
            # Create x values for the interpolated data
            x_interp = np.linspace(0, 1, max_length)
            
            # Create cubic spline
            cs = CubicSpline(x_orig, col_data)
            
            # Interpolate the data
            interp_data[col_name] = cs(x_interp)
        
        # Resample to exactly num_samples (default: 149) steps
        resampled_data = {}
        for col_name, col_data in interp_data.items():
            # Create x values for the current interpolated data
            x_current = np.linspace(0, 1, len(col_data))
            
            # Create x values for the target number of samples
            x_target = np.linspace(0, 1, num_samples)
            
            # Create cubic spline for resampling
            cs_resample = CubicSpline(x_current, col_data)
            
            # Resample the data
            resampled_data[col_name] = cs_resample(x_target)
        
        # Create DataFrame from resampled data
        df_output = pd.DataFrame(resampled_data)
        
        # Save to Excel
        df_output.to_excel(output_file, index=False)
        print(f"Data extracted and resampled to {num_samples} steps, saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract columns from Excel file, interpolate and save to a new file.")
    parser.add_argument("input_file", help="Path to input Excel file")
    parser.add_argument("output_file", help="Path to output Excel file")
    parser.add_argument("--columns", nargs="+", help="List of sheet:column pairs to extract (e.g., 'Sheet1:Column1')")
    parser.add_argument("--samples", type=int, default=149, help="Number of samples to resample the data to (default: 149)")
    
    args = parser.parse_args()
    
    # Call the extraction function
    success = extract_from_excel(args.input_file, args.output_file, args.columns, args.samples)
    
    # Exit with appropriate status
    sys.exit(0 if success else 1)