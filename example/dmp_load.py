import numpy as np
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up workspace path and import pymplibrary
current_file = os.path.abspath(__file__)
if '/install/' in current_file:
    ws_path = current_file[:current_file.find('/install/')]
elif '/src/' in current_file:
    ws_path = current_file[:current_file.find('/src/')]
else:
    logger.error("Could not determine workspace path. Script must be run from install or src directory.")
    sys.exit(1)

mplibrary_path = os.path.join(ws_path, 'build/mplibrary')
sys.path.append(mplibrary_path)
import pymplibrary as motion
    
def main():
    # Create output directory
    output_dir = os.path.join(ws_path, "src/manipulator_skill_acquisition/resources/dmp")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    output_file = open(os.path.join(output_dir, "dmp_trajectory.txt"), "w")
    
    # Load DMP from MPX file
    mpx_path = os.path.join(ws_path, "src/manipulator_skill_acquisition/resources/dmp/dmp.mpx")
    library = motion.mpx.load_from(mpx_path)
    policy = library.policies[0]
    phase = motion.LinearPacer(1.0)
    ts = 0.01

    # First pass to generate trajectory
    phase.value = 0.0
    # Set initial joint positions (replace with your desired starting position)
    policy.reset([0.0, 0.0, 0.0])
    
    # Write initial position
    out = f"{policy.value[0]};{policy.value[1]};{policy.value[2]}\n"
    output_file.write(out)
    logger.info("Starting trajectory generation")
    
    # Generate trajectory
    while phase.value < 0.999:
        phase.update(ts)
        policy.update(ts, phase.value)
        
        # Write to file
        out = f"{policy.value[0]};{policy.value[1]};{policy.value[2]}\n"
        output_file.write(out)
        
    # Close file
    output_file.close()
    logger.info("DMP trajectory generation completed")

if __name__ == "__main__":
    main()
