#!/usr/bin/python3

# import math
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pkg_resources
from ament_index_python import get_package_share_directory

# Get workspace path from current file
current_file = os.path.abspath(__file__)
if '/install/' in current_file:
    ws_path = current_file[:current_file.find('/install/')]
elif '/src/' in current_file:
    ws_path = current_file[:current_file.find('/src/')]
else:
    print("Error: Could not determine workspace path. Script must be run from install or src directory.")
    sys.exit(1)

# Add build directories to path
mplibrary_path = os.path.join(ws_path, 'build/mplibrary')
mplearn_path = os.path.join(ws_path, 'build/mplearn')

sys.path.append(mplibrary_path)
import pymplibrary as motion

sys.path.append(mplearn_path)
import pymplearn as learn

def print_usage():
    print("""
DMP Generator - Generates Dynamic Movement Primitives from reference trajectories

Usage:
    python dmp_generator.py <config_file> <reference_file> <output_file>

Arguments:
    config_file     : Full path to the DMP configuration YAML file
    reference_file  : Path to the reference trajectory CSV file
    output_file     : Path where the generated DMP will be saved

Example:
    python dmp_generator.py /path/to/config/discrete.yaml data/trajectory.csv output/dmp.mpx

The script will:
1. Load the DMP configuration from the YAML file
2. Read the reference trajectory from the CSV file
3. Generate a DMP that matches the reference trajectory
4. Save the DMP to the specified output file

Output:
- The generated DMP will be saved as a .mpx file
- Progress information will be displayed during generation
- A success message will be shown when complete
""")

def validate_files(config_path, reference_path, save_path):
    """Validate input and output files"""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return False
    
    if not os.path.exists(reference_path):
        print(f"Error: Reference file not found: {reference_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return True

if len(sys.argv) < 4:
    print_usage()
    sys.exit(1)

config_path = sys.argv[1]
reference_path = sys.argv[2]
save_path = sys.argv[3]

# Validate files
if not validate_files(config_path, reference_path, save_path):
    sys.exit(1)

print(f"\nStarting DMP generation...")
print(f"Config file: {config_path}")
print(f"Reference file: {reference_path}")
print(f"Output file: {save_path}\n")

# load DMP learning configuration
try:
    print("Loading DMP configuration...")
    config = learn.regression.Config.load(config_path)
    print(f"Configuration loaded successfully")
    print(f"Number of basis functions: {config.n_basis}")
    print(f"Number of primitives: {config.n_primitives}")
except Exception as err:
    print(f"Error loading configuration: {err}")
    sys.exit(1)

# load reference data to learn
print("\nLoading reference trajectory...")
try:
    reference = np.loadtxt(reference_path, delimiter=',')
    n_samples, n_dims = reference.shape
    print(f"Reference trajectory loaded: {n_samples} samples, {n_dims} dimensions")
except Exception as err:
    print(f"Error loading reference trajectory: {err}")
    sys.exit(1)

# assign phase values
print("\nGenerating phase values...")
pvals = [0.0] * n_samples
tspan = n_samples * config.rollout.integration_timestep
phase = motion.LinearPacer(1.0)
phase.limits.lower = config.phase_bounds[0]
phase.limits.upper = config.phase_bounds[1]
phase.pace = (phase.limits.upper - phase.limits.lower) / tspan
phase.value = phase.limits.lower
for idx in range(n_samples):
    pvals[idx] = phase.value
    phase.update(config.rollout.integration_timestep)

print(f"Phase range: [{pvals[0]:.3f} ... {pvals[-1]:.3f}]")

# initialize DMP system
print("\nInitializing DMP system...")
library = motion.Library()
library.primitives = motion.Primitive.withGaussianBasis(
    config.n_primitives, 
    config.n_basis, 
    phase.limits.lower, 
    phase.limits.upper, 
    motion.kernel.Gaussian.scaleWidth(config.n_basis)
)

# initialize skill and policy
print("Setting up DMP policy...")
skill = motion.Skill(library.primitives, n_dims)
skill.setWeights(motion.mat(np.identity(n_dims)))

policy = motion.DMP(n_dims, skill)

# parametrize DMP
print("Configuring DMP parameters...")
for idx, dim in enumerate(policy):
    dim.goal = learn.regression.getDMPBaseline(reference[:, idx], config.baseline_method)
    dim.parameters = config.damping
    dim.value = reference[0, idx]

# run optimization
print("\nRunning DMP optimization...")
sols = learn.regression.fitPolicy(motion.mat(reference.transpose()), pvals, policy, "", config)
print("Optimization complete")

# normalize DMP temporal scale
for dim in policy:
    dim.tau = 1.0 / phase.pace

# export to file
print("\nSaving DMP...")
library.skills.add(skill)
library.policies.add(policy)
motion.mpx.save(library, save_path)

print(f"""
DMP Generation Complete!
------------------------
Output file: {save_path}
File size: {os.path.getsize(save_path) / 1024:.2f} KB

The DMP has been successfully generated and saved.
You can now use this DMP file for motion planning and execution.
""")