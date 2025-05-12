import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging
import json
from datetime import datetime
from scipy.spatial import procrustes

# Set up logging to both file and console
log_dir = Path(__file__).parent / 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / f'velocity_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def save_progress(participant, trial, file_type):
    """Save progress to a JSON file."""
    progress_file = Path(__file__).parent / 'data' / 'analysis_progress.json'
    progress = {}
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except:
            pass
    
    if participant not in progress:
        progress[participant] = {}
    if trial not in progress[participant]:
        progress[participant][trial] = []
    
    if file_type not in progress[participant][trial]:
        progress[participant][trial].append(file_type)
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress():
    """Load progress from JSON file."""
    progress_file = Path(__file__).parent / 'data' / 'analysis_progress.json'
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def calculate_joint_correlations(df, dt=1):
    """Calculate correlations between joint velocities."""
    # Calculate velocities for all joints
    velocities = {}
    for i in range(1, 4):
        traj1_col = f'traj1_ang{i}'
        traj2_col = f'traj2_ang{i}'
        if traj1_col in df.columns and traj2_col in df.columns:
            velocities[f'Traj1_Joint{i}'] = df[traj1_col].diff()/dt
            velocities[f'Traj2_Joint{i}'] = df[traj2_col].diff()/dt
    
    # Create DataFrame for correlations
    vel_df = pd.DataFrame(velocities)
    
    # Calculate correlations for each trajectory
    traj1_corr = vel_df[[f'Traj1_Joint{i}' for i in range(1, 4)]].corr()
    traj2_corr = vel_df[[f'Traj2_Joint{i}' for i in range(1, 4)]].corr()
    
    return traj1_corr, traj2_corr

def plot_joint_correlations(traj1_corr, traj2_corr, participant, trial, output_dir):
    """Plot correlation matrices for both trajectories."""
    plt.figure(figsize=(12, 5))
    
    # Create masks for upper triangle (excluding diagonal)
    mask1 = np.triu(np.ones_like(traj1_corr, dtype=bool), k=1)
    mask2 = np.triu(np.ones_like(traj2_corr, dtype=bool), k=1)
    
    # Create a single heatmap with two subplots
    plt.subplot(1, 2, 1)
    sns.heatmap(traj1_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=['J1', 'J2', 'J3'],
                yticklabels=['J1', 'J2', 'J3'],
                cbar_kws={'label': 'Correlation'},
                mask=~mask1)  # Show only upper triangle
    plt.title('Traj1\nJoint Correlations')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(traj2_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=['J1', 'J2', 'J3'],
                yticklabels=['J1', 'J2', 'J3'],
                cbar_kws={'label': 'Correlation'},
                mask=~mask2)  # Show only upper triangle
    plt.title('Traj2\nJoint Correlations')
    
    plt.suptitle(f'Joint Velocity Correlations - {participant} - Trial {trial}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'joint_correlations_{participant}_trial_{trial}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print concise correlation summary
    logger.info(f"\nJoint Correlation Summary for {participant} - Trial {trial}:")
    
    # Get upper triangle correlations (excluding diagonal)
    traj1_upper = traj1_corr.where(mask1).unstack()
    traj2_upper = traj2_corr.where(mask2).unstack()
    
    # Remove NaN values and sort
    traj1_strong = traj1_upper.dropna().sort_values(ascending=False)
    traj2_strong = traj2_upper.dropna().sort_values(ascending=False)
    
    logger.info("\nStrongest correlations:")
    logger.info("Traj1:")
    for (j1, j2), corr in traj1_strong.head(3).items():
        logger.info(f"  {j1}-{j2}: {corr:.3f}")
    logger.info("Traj2:")
    for (j1, j2), corr in traj2_strong.head(3).items():
        logger.info(f"  {j1}-{j2}: {corr:.3f}")

def process_single_file(file_path, file_type, participant, trial):
    """Process a single file and generate visualizations."""
    try:
        logger.info(f"Processing {file_type} file: {file_path}")
        
        if file_type == 'ee':
            # Process end-effector data
            df = pd.read_excel(file_path, skiprows=3, usecols=['Traj1_X', 'Traj1_Z', 'Traj2_X', 'Traj2_Z'])
            
            # Calculate velocities
            dt = 1
            velocity_traj1 = np.sqrt((df['Traj1_X'].diff()/dt)**2 + (df['Traj1_Z'].diff()/dt)**2)
            velocity_traj2 = np.sqrt((df['Traj2_X'].diff()/dt)**2 + (df['Traj2_Z'].diff()/dt)**2)
            
            # Generate visualization
            output_dir = Path(__file__).parent / 'data' / 'visualizations_velocity_profiles'
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(15, 6))
            plt.plot(velocity_traj1, label='Traj1 Velocity', color='blue')
            plt.plot(velocity_traj2, label='Traj2 Velocity', color='red')
            plt.title(f'End-Effector Velocity Profile - {participant} - Trial {trial}')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity (mm/s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f'ee_velocity_{participant}_trial_{trial}.png')
            plt.close()
            
            # Print statistics
            logger.info(f"\nEnd-Effector Velocity Statistics for {participant} - Trial {trial}:")
            logger.info(f"  Traj1 - Mean Velocity: {float(velocity_traj1.mean()):.4f} mm/s")
            logger.info(f"  Traj1 - Velocity Consistency: {float(velocity_traj1.std()):.4f} mm/s")
            logger.info(f"  Traj2 - Mean Velocity: {float(velocity_traj2.mean()):.4f} mm/s")
            logger.info(f"  Traj2 - Velocity Consistency: {float(velocity_traj2.std()):.4f} mm/s")
            
        elif file_type == 'ang':
            # Process joint data
            columns = [f'traj1_ang{i}' for i in range(1, 4)] + [f'traj2_ang{i}' for i in range(1, 4)]
            df = pd.read_excel(file_path, skiprows=5, usecols=columns)
            
            # Calculate joint velocities
            dt = 1
            output_dir = Path(__file__).parent / 'data' / 'visualizations_velocity_profiles'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a single figure for all joints
            plt.figure(figsize=(15, 10))
            
            # Plot each joint in a subplot
            for i, joint in enumerate(['Joint1', 'Joint2', 'Joint3'], 1):
                traj1_col = f'traj1_ang{joint[-1]}'
                traj2_col = f'traj2_ang{joint[-1]}'
                
                if traj1_col in df.columns and traj2_col in df.columns:
                    velocity_traj1 = df[traj1_col].diff()/dt
                    velocity_traj2 = df[traj2_col].diff()/dt
                    
                    plt.subplot(3, 1, i)
                    plt.plot(velocity_traj1, label=f'Traj1 {joint}', color='blue')
                    plt.plot(velocity_traj2, label=f'Traj2 {joint}', color='red')
                    plt.title(f'{joint} Velocity Profile')
                    plt.xlabel('Time Step')
                    plt.ylabel('Velocity (degrees/s)')
                    plt.legend()
                    plt.grid(True)
            
            plt.suptitle(f'Joint Velocity Profiles - {participant} - Trial {trial}', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / f'joint_velocity_{participant}_trial_{trial}.png')
            plt.close()
            
            # Calculate and plot joint correlations
            traj1_corr, traj2_corr = calculate_joint_correlations(df, dt)
            plot_joint_correlations(traj1_corr, traj2_corr, participant, trial, output_dir)
            
            # Print statistics for each joint
            for joint in ['Joint1', 'Joint2', 'Joint3']:
                traj1_col = f'traj1_ang{joint[-1]}'
                traj2_col = f'traj2_ang{joint[-1]}'
                
                if traj1_col in df.columns and traj2_col in df.columns:
                    velocity_traj1 = df[traj1_col].diff()/dt
                    velocity_traj2 = df[traj2_col].diff()/dt
                    
                    logger.info(f"\n{joint} Velocity Statistics for {participant} - Trial {trial}:")
                    logger.info(f"  Traj1 - Mean Velocity: {float(velocity_traj1.mean()):.4f} degrees/s")
                    logger.info(f"  Traj1 - Velocity Consistency: {float(velocity_traj1.std()):.4f} degrees/s")
                    logger.info(f"  Traj2 - Mean Velocity: {float(velocity_traj2.mean()):.4f} degrees/s")
                    logger.info(f"  Traj2 - Velocity Consistency: {float(velocity_traj2.std()):.4f} degrees/s")
            
            # Print correlation statistics
            logger.info(f"\nJoint Correlation Statistics for {participant} - Trial {trial}:")
            logger.info("\nTraj1 Joint Correlations:")
            logger.info(traj1_corr.to_string())
            logger.info("\nTraj2 Joint Correlations:")
            logger.info(traj2_corr.to_string())
        
        # Save progress
        save_progress(participant, trial, file_type)
        
        # Clear memory
        del df
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_type} file {file_path}: {str(e)}")
        return False

def main():
    # Get the absolute path to the workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    base_path = workspace_root / 'src' / 'manipulator_skill_acquisition' / 'resources' / 'dados_excel'
    output_dir = Path(__file__).parent / 'data' / 'visualizations_joint_correlations_velocity_profiles'
    
    logger.info("\n" + "="*50)
    logger.info("MANIPULATOR SKILL ACQUISITION - VELOCITY PROFILE ANALYSIS")
    logger.info("="*50)
    
    # Load previous progress
    progress = load_progress()
    
    participants = ['adriana', 'afonsoleite', 'carolinamaia', 'catarina', 
                   'diogo', 'jorge', 'luis', 'sara', 'vitor']
    
    for participant_idx, participant in enumerate(participants, 1):
        logger.info(f"\nProcessing participant {participant_idx}/{len(participants)}: {participant}")
        participant_path = base_path / participant / 'results'
        
        if participant_path.exists():
            for trial in range(1, 9):
                logger.info(f"\nProcessing trial {trial}/8")
                
                # Skip if already processed
                if (participant in progress and 
                    str(trial) in progress[participant] and 
                    len(progress[participant][str(trial)]) == 2):
                    logger.info(f"Skipping already processed trial {trial}")
                    continue
                
                # Process end-effector data
                ee_file_patterns = [
                    f'ee_error_{trial}.xlsx',
                    f'ee_error_{trial:02d}.xlsx',
                    f'ee_error_{trial}.xls',
                    f'ee_error_{trial:02d}.xls',
                    f'ee_{trial}.xlsx',
                    f'ee_{trial:02d}.xlsx'
                ]
                
                for pattern in ee_file_patterns:
                    file_path = participant_path / pattern
                    if file_path.exists():
                        process_single_file(file_path, 'ee', participant, trial)
                        break
                
                # Process joint data
                ang_file_patterns = [
                    f'ang_error_{trial}.xlsx',
                    f'ang_error_{trial:02d}.xlsx',
                    f'ang_error_{trial}.xls',
                    f'ang_error_{trial:02d}.xls',
                    f'ang_{trial}.xlsx',
                    f'ang_{trial:02d}.xlsx'
                ]
                
                for pattern in ang_file_patterns:
                    file_path = participant_path / pattern
                    if file_path.exists():
                        process_single_file(file_path, 'ang', participant, trial)
                        break
                
                # Force garbage collection after each trial
                import gc
                gc.collect()
        
        # Force garbage collection after each participant
        import gc
        gc.collect()
    
    logger.info("\n" + "="*50)
    logger.info("Analysis complete. Visualizations have been saved to the 'visualizations_velocity_profiles' directory")
    logger.info("="*50)

if __name__ == "__main__":
    main() 