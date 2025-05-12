import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def load_all_data(base_path):
    """Load all Excel files from the results directory."""
    participants = ['adriana', 'afonsoleite', 'carolinamaia', 'catarina', 
                   'diogo', 'jorge', 'luis', 'sara', 'vitor']
    
    ee_data = []
    ang_data = []
    
    # Convert base_path to absolute path
    base_path = Path(base_path).absolute()
    logger.info(f"\nBase path: {base_path}")
    logger.info(f"Base path exists: {base_path.exists()}")
    
    if base_path.exists():
        logger.info(f"Contents of base path: {list(base_path.glob('*'))}")
    else:
        logger.error(f"Base path does not exist: {base_path}")
        raise ValueError(f"Base path does not exist: {base_path}")
    
    for participant in participants:
        participant_path = base_path / participant / 'results'
        logger.info(f"\nProcessing participant: {participant}")
        logger.info(f"Checking path: {participant_path}")
        logger.info(f"Path exists: {participant_path.exists()}")
        
        if participant_path.exists():
            # List all files in the participant directory
            all_files = list(participant_path.glob('*'))
            logger.info(f"All files in {participant} directory: {all_files}")
            
            # Load end-effector data
            for i in range(1, 9):
                # Try different possible file patterns
                file_patterns = [
                    f'ee_error_{i}.xlsx',
                    f'ee_error_{i:02d}.xlsx',
                    f'ee_error_{i}.xls',
                    f'ee_error_{i:02d}.xls',
                    f'ee_{i}.xlsx',
                    f'ee_{i:02d}.xlsx'
                ]
                
                file_found = False
                for pattern in file_patterns:
                    file_path = participant_path / pattern
                    logger.info(f"Trying file pattern: {pattern}")
                    logger.info(f"File exists: {file_path.exists()}")
                    
                    if file_path.exists():
                        try:
                            # Skip the summary rows and read the trajectory data
                            df = pd.read_excel(file_path, skiprows=3)  # Skip summary and blank row
                            logger.info(f"Successfully loaded trajectory data from {file_path}")
                            logger.info(f"Columns found: {df.columns.tolist()}")
                            
                            # Calculate error as Euclidean distance between trajectories
                            error = np.sqrt(
                                (df['Traj1_X'] - df['Traj2_X'])**2 + 
                                (df['Traj1_Z'] - df['Traj2_Z'])**2
                            )
                            
                            # Calculate velocities and accelerations
                            dt = 1  # Assuming constant time step
                            velocity_traj1 = np.sqrt(
                                (df['Traj1_X'].diff()/dt)**2 + 
                                (df['Traj1_Z'].diff()/dt)**2
                            )
                            velocity_traj2 = np.sqrt(
                                (df['Traj2_X'].diff()/dt)**2 + 
                                (df['Traj2_Z'].diff()/dt)**2
                            )
                            
                            acceleration_traj1 = velocity_traj1.diff()/dt
                            acceleration_traj2 = velocity_traj2.diff()/dt
                            
                            # Calculate jerk (rate of change of acceleration)
                            jerk_traj1 = acceleration_traj1.diff()/dt
                            jerk_traj2 = acceleration_traj2.diff()/dt
                            
                            # Calculate path length
                            path_length_traj1 = np.sum(np.sqrt(
                                (df['Traj1_X'].diff())**2 + 
                                (df['Traj1_Z'].diff())**2
                            ))
                            path_length_traj2 = np.sum(np.sqrt(
                                (df['Traj2_X'].diff())**2 + 
                                (df['Traj2_Z'].diff())**2
                            ))
                            
                            # Calculate comprehensive statistics from trajectory data
                            trajectory_stats = {
                                # Basic error metrics
                                'Mean_Error': error.mean(),
                                'Median_Error': error.median(),
                                'Std_Error': error.std(),
                                'Min_Error': error.min(),
                                'Max_Error': error.max(),
                                'Range_Error': error.max() - error.min(),
                                
                                # Quartile statistics
                                'Q1_Error': error.quantile(0.25),
                                'Q3_Error': error.quantile(0.75),
                                'IQR_Error': error.quantile(0.75) - error.quantile(0.25),
                                
                                # Distribution statistics
                                'Skewness': error.skew(),
                                'Kurtosis': error.kurtosis(),
                                
                                # Error metrics
                                'RMSE': np.sqrt(np.mean(error**2)),
                                'MAE': np.mean(np.abs(error)),
                                
                                # Time-based metrics
                                'Duration': len(error) * dt,
                                'Error_Integral': np.trapz(error, dx=dt),
                                'Error_Variance': np.var(error),
                                
                                # Velocity metrics
                                'Mean_Velocity_Traj1': velocity_traj1.mean(),
                                'Mean_Velocity_Traj2': velocity_traj2.mean(),
                                'Velocity_RMSE': np.sqrt(np.mean((velocity_traj1 - velocity_traj2)**2)),
                                'Velocity_Correlation': velocity_traj1.corr(velocity_traj2),
                                
                                # Acceleration metrics
                                'Mean_Acceleration_Traj1': acceleration_traj1.mean(),
                                'Mean_Acceleration_Traj2': acceleration_traj2.mean(),
                                'Acceleration_RMSE': np.sqrt(np.mean((acceleration_traj1 - acceleration_traj2)**2)),
                                'Acceleration_Correlation': acceleration_traj1.corr(acceleration_traj2),
                                
                                # Jerk metrics
                                'Mean_Jerk_Traj1': jerk_traj1.mean(),
                                'Mean_Jerk_Traj2': jerk_traj2.mean(),
                                'Jerk_RMSE': np.sqrt(np.mean((jerk_traj1 - jerk_traj2)**2)),
                                
                                # Path metrics
                                'Path_Length_Traj1': path_length_traj1,
                                'Path_Length_Traj2': path_length_traj2,
                                'Path_Length_Ratio': path_length_traj2 / path_length_traj1,
                                
                                # Additional statistical measures
                                'Error_Entropy': -np.sum(error * np.log2(error + 1e-10)),
                                'Error_Autocorrelation': error.autocorr(),
                                'Error_Stationarity': np.mean(np.abs(error.diff())),
                                
                                'participant': participant,
                                'trial': i,
                                'error_type': 'end_effector'
                            }
                            ee_data.append(pd.DataFrame([trajectory_stats]))
                            logger.info(f"Added data for participant {participant}, trial {i}")
                            file_found = True
                            break
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {str(e)}")
                            logger.error(f"Error type: {type(e)}")
                            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
                
                if not file_found:
                    logger.warning(f"No end-effector file found for participant {participant}, trial {i}")
            
            # Load angular data
            for i in range(1, 9):
                # Try different possible file patterns
                file_patterns = [
                    f'ang_error_{i}.xlsx',
                    f'ang_error_{i:02d}.xlsx',
                    f'ang_error_{i}.xls',
                    f'ang_error_{i:02d}.xls',
                    f'ang_{i}.xlsx',
                    f'ang_{i:02d}.xlsx'
                ]
                
                file_found = False
                for pattern in file_patterns:
                    file_path = participant_path / pattern
                    logger.info(f"Trying file pattern: {pattern}")
                    logger.info(f"File exists: {file_path.exists()}")
                    
                    if file_path.exists():
                        try:
                            # Skip the summary rows and read the trajectory data
                            df = pd.read_excel(file_path, skiprows=5)  # Skip summary and blank row
                            logger.info(f"Successfully loaded trajectory data from {file_path}")
                            logger.info(f"Columns found: {df.columns.tolist()}")
                            
                            # Calculate error for each joint
                            for joint in ['Joint1', 'Joint2', 'Joint3']:
                                traj1_col = f'traj1_ang{joint[-1]}'
                                traj2_col = f'traj2_ang{joint[-1]}'
                                
                                if traj1_col in df.columns and traj2_col in df.columns:
                                    error = abs(df[traj1_col] - df[traj2_col])
                                    
                                    # Calculate velocities and accelerations
                                    dt = 1  # Assuming constant time step
                                    velocity_traj1 = df[traj1_col].diff()/dt
                                    velocity_traj2 = df[traj2_col].diff()/dt
                                    
                                    acceleration_traj1 = velocity_traj1.diff()/dt
                                    acceleration_traj2 = velocity_traj2.diff()/dt
                                    
                                    # Calculate jerk
                                    jerk_traj1 = acceleration_traj1.diff()/dt
                                    jerk_traj2 = acceleration_traj2.diff()/dt
                                    
                                    # Calculate path length (total angular displacement)
                                    path_length_traj1 = np.sum(np.abs(df[traj1_col].diff()))
                                    path_length_traj2 = np.sum(np.abs(df[traj2_col].diff()))
                                    
                                    # Calculate comprehensive statistics from trajectory data
                                    trajectory_stats = {
                                        # Basic error metrics
                                        'Mean_Error': error.mean(),
                                        'Median_Error': error.median(),
                                        'Std_Error': error.std(),
                                        'Min_Error': error.min(),
                                        'Max_Error': error.max(),
                                        'Range_Error': error.max() - error.min(),
                                        
                                        # Quartile statistics
                                        'Q1_Error': error.quantile(0.25),
                                        'Q3_Error': error.quantile(0.75),
                                        'IQR_Error': error.quantile(0.75) - error.quantile(0.25),
                                        
                                        # Distribution statistics
                                        'Skewness': error.skew(),
                                        'Kurtosis': error.kurtosis(),
                                        
                                        # Error metrics
                                        'RMSE': np.sqrt(np.mean(error**2)),
                                        'MAE': np.mean(np.abs(error)),
                                        
                                        # Time-based metrics
                                        'Duration': len(error) * dt,
                                        'Error_Integral': np.trapz(error, dx=dt),
                                        'Error_Variance': np.var(error),
                                        
                                        # Velocity metrics
                                        'Mean_Velocity_Traj1': velocity_traj1.mean(),
                                        'Mean_Velocity_Traj2': velocity_traj2.mean(),
                                        'Velocity_RMSE': np.sqrt(np.mean((velocity_traj1 - velocity_traj2)**2)),
                                        'Velocity_Correlation': velocity_traj1.corr(velocity_traj2),
                                        
                                        # Acceleration metrics
                                        'Mean_Acceleration_Traj1': acceleration_traj1.mean(),
                                        'Mean_Acceleration_Traj2': acceleration_traj2.mean(),
                                        'Acceleration_RMSE': np.sqrt(np.mean((acceleration_traj1 - acceleration_traj2)**2)),
                                        'Acceleration_Correlation': acceleration_traj1.corr(acceleration_traj2),
                                        
                                        # Jerk metrics
                                        'Mean_Jerk_Traj1': jerk_traj1.mean(),
                                        'Mean_Jerk_Traj2': jerk_traj2.mean(),
                                        'Jerk_RMSE': np.sqrt(np.mean((jerk_traj1 - jerk_traj2)**2)),
                                        
                                        # Path metrics
                                        'Path_Length_Traj1': path_length_traj1,
                                        'Path_Length_Traj2': path_length_traj2,
                                        'Path_Length_Ratio': path_length_traj2 / path_length_traj1,
                                        
                                        # Additional statistical measures
                                        'Error_Entropy': -np.sum(error * np.log2(error + 1e-10)),
                                        'Error_Autocorrelation': error.autocorr(),
                                        'Error_Stationarity': np.mean(np.abs(error.diff())),
                                        
                                        'Joint': joint,
                                        'participant': participant,
                                        'trial': i,
                                        'error_type': 'angular'
                                    }
                                    ang_data.append(pd.DataFrame([trajectory_stats]))
                                    logger.info(f"Added data for participant {participant}, trial {i}, joint {joint}")
                            file_found = True
                            break
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {str(e)}")
                            logger.error(f"Error type: {type(e)}")
                            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
                
                if not file_found:
                    logger.warning(f"No angular file found for participant {participant}, trial {i}")
    
    logger.info(f"\nLoaded {len(ee_data)} end-effector files and {len(ang_data)} angular files")
    
    if not ee_data and not ang_data:
        logger.error("No data files were successfully loaded.")
        logger.error(f"Base path: {base_path}")
        logger.error(f"Base path exists: {base_path.exists()}")
        if base_path.exists():
            logger.error(f"Contents of base path: {list(base_path.glob('*'))}")
        raise ValueError("No data files were successfully loaded.")
    
    ee_df = pd.concat(ee_data) if ee_data else pd.DataFrame()
    ang_df = pd.concat(ang_data) if ang_data else pd.DataFrame()
    
    # Ensure all error values are positive
    if not ee_df.empty:
        for col in ee_df.columns:
            if 'Error' in col or 'RMSE' in col or 'MAE' in col:
                ee_df[col] = ee_df[col].abs()
    
    if not ang_df.empty:
        for col in ang_df.columns:
            if 'Error' in col or 'RMSE' in col or 'MAE' in col:
                ang_df[col] = ang_df[col].abs()
    
    logger.info(f"End-effector data shape: {ee_df.shape if not ee_df.empty else 'empty'}")
    logger.info(f"Angular data shape: {ang_df.shape if not ang_df.empty else 'empty'}")
    
    return ee_df, ang_df

def generate_visualizations(ee_data, ang_data, output_dir):
    """Generate visualizations for all metrics."""
    logger.info(f"\nGenerating visualizations in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    plt.style.use('seaborn')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    if not ee_data.empty:
        logger.info("Processing end-effector data visualizations")
        
        # 1. Basic Error Metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic Error Metrics by Participant', y=1.02)
        
        # Mean Error
        sns.boxplot(x='participant', y='Mean_Error', data=ee_data, ax=axes[0,0])
        axes[0,0].set_title('Mean Position Error')
        axes[0,0].set_ylabel('Error (mm)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Median Error
        sns.boxplot(x='participant', y='Median_Error', data=ee_data, ax=axes[0,1])
        axes[0,1].set_title('Median Position Error')
        axes[0,1].set_ylabel('Error (mm)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Range Error
        sns.boxplot(x='participant', y='Range_Error', data=ee_data, ax=axes[1,0])
        axes[1,0].set_title('Position Error Range')
        axes[1,0].set_ylabel('Error (mm)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # IQR Error
        sns.boxplot(x='participant', y='IQR_Error', data=ee_data, ax=axes[1,1])
        axes[1,1].set_title('Position Error IQR')
        axes[1,1].set_ylabel('Error (mm)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_basic_error_metrics.png'))
        plt.close()
        
        # 2. Distribution Metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Error Distribution Metrics by Participant', y=1.02)
        
        # Skewness
        sns.boxplot(x='participant', y='Skewness', data=ee_data, ax=axes[0])
        axes[0].set_title('Error Distribution Skewness')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Kurtosis
        sns.boxplot(x='participant', y='Kurtosis', data=ee_data, ax=axes[1])
        axes[1].set_title('Error Distribution Kurtosis')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_distribution_metrics.png'))
        plt.close()
        
        # 3. Time-based Metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time-based Metrics by Participant', y=1.02)
        
        # Duration
        sns.boxplot(x='participant', y='Duration', data=ee_data, ax=axes[0,0])
        axes[0,0].set_title('Movement Duration')
        axes[0,0].set_ylabel('Time (s)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Error Integral
        sns.boxplot(x='participant', y='Error_Integral', data=ee_data, ax=axes[0,1])
        axes[0,1].set_title('Cumulative Error')
        axes[0,1].set_ylabel('Error × Time (mm·s)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Error Variance
        sns.boxplot(x='participant', y='Error_Variance', data=ee_data, ax=axes[1,0])
        axes[1,0].set_title('Error Variance')
        axes[1,0].set_ylabel('Variance (mm²)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Error Stationarity
        sns.boxplot(x='participant', y='Error_Stationarity', data=ee_data, ax=axes[1,1])
        axes[1,1].set_title('Error Stationarity')
        axes[1,1].set_ylabel('Mean Absolute Change')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_time_based_metrics.png'))
        plt.close()
        
        # 4. Kinematic Metrics
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Kinematic Metrics by Participant', y=1.02)
        
        # Velocity
        sns.boxplot(x='participant', y='Mean_Velocity_Traj1', data=ee_data, ax=axes[0,0])
        axes[0,0].set_title('Mean Velocity (Traj1)')
        axes[0,0].set_ylabel('Velocity (mm/s)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x='participant', y='Mean_Velocity_Traj2', data=ee_data, ax=axes[0,1])
        axes[0,1].set_title('Mean Velocity (Traj2)')
        axes[0,1].set_ylabel('Velocity (mm/s)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Acceleration
        sns.boxplot(x='participant', y='Mean_Acceleration_Traj1', data=ee_data, ax=axes[1,0])
        axes[1,0].set_title('Mean Acceleration (Traj1)')
        axes[1,0].set_ylabel('Acceleration (mm/s²)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x='participant', y='Mean_Acceleration_Traj2', data=ee_data, ax=axes[1,1])
        axes[1,1].set_title('Mean Acceleration (Traj2)')
        axes[1,1].set_ylabel('Acceleration (mm/s²)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Jerk
        sns.boxplot(x='participant', y='Mean_Jerk_Traj1', data=ee_data, ax=axes[2,0])
        axes[2,0].set_title('Mean Jerk (Traj1)')
        axes[2,0].set_ylabel('Jerk (mm/s³)')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x='participant', y='Mean_Jerk_Traj2', data=ee_data, ax=axes[2,1])
        axes[2,1].set_title('Mean Jerk (Traj2)')
        axes[2,1].set_ylabel('Jerk (mm/s³)')
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_kinematic_metrics.png'))
        plt.close()
        
        # 5. Path Metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Path Metrics by Participant', y=1.02)
        
        # Path Length
        sns.boxplot(x='participant', y='Path_Length_Traj1', data=ee_data, ax=axes[0])
        axes[0].set_title('Path Length (Traj1)')
        axes[0].set_ylabel('Length (mm)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Path Length Ratio
        sns.boxplot(x='participant', y='Path_Length_Ratio', data=ee_data, ax=axes[1])
        axes[1].set_title('Path Length Ratio (Traj2/Traj1)')
        axes[1].set_ylabel('Ratio')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_path_metrics.png'))
        plt.close()
        
        # 6. Advanced Statistical Metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Advanced Statistical Metrics by Participant', y=1.02)
        
        # Error Entropy
        sns.boxplot(x='participant', y='Error_Entropy', data=ee_data, ax=axes[0])
        axes[0].set_title('Error Entropy')
        axes[0].set_ylabel('Entropy (bits)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Error Autocorrelation
        sns.boxplot(x='participant', y='Error_Autocorrelation', data=ee_data, ax=axes[1])
        axes[1].set_title('Error Autocorrelation')
        axes[1].set_ylabel('Correlation')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_advanced_metrics.png'))
        plt.close()
        
        # Create heatmap of position errors
        fig, ax = plt.subplots(figsize=(12, 8))  # Restored original horizontal orientation
        heatmap_data = ee_data.pivot_table(
            values='Mean_Error',
            index='participant',
            columns='trial',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
        ax.set_title('End-Effector Position Error Heatmap (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ee_position_error_heatmap.png'))
        plt.close()
    
    if not ang_data.empty:
        logger.info("Processing angular data visualizations")
        
        # Similar visualizations for angular data, but organized by joint
        for joint in ang_data['Joint'].unique():
            joint_data = ang_data[ang_data['Joint'] == joint]
            
            # 1. Basic Error Metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Basic Error Metrics by Participant - {joint}', y=1.02)
            
            # Mean Error
            sns.boxplot(x='participant', y='Mean_Error', data=joint_data, ax=axes[0,0])
            axes[0,0].set_title('Mean Angle Error')
            axes[0,0].set_ylabel('Error (degrees)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Median Error
            sns.boxplot(x='participant', y='Median_Error', data=joint_data, ax=axes[0,1])
            axes[0,1].set_title('Median Angle Error')
            axes[0,1].set_ylabel('Error (degrees)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Range Error
            sns.boxplot(x='participant', y='Range_Error', data=joint_data, ax=axes[1,0])
            axes[1,0].set_title('Angle Error Range')
            axes[1,0].set_ylabel('Error (degrees)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # IQR Error
            sns.boxplot(x='participant', y='IQR_Error', data=joint_data, ax=axes[1,1])
            axes[1,1].set_title('Angle Error IQR')
            axes[1,1].set_ylabel('Error (degrees)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ang_{joint}_basic_error_metrics.png'))
            plt.close()
            
            # 2. Kinematic Metrics
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle(f'Kinematic Metrics by Participant - {joint}', y=1.02)
            
            # Velocity
            sns.boxplot(x='participant', y='Mean_Velocity_Traj1', data=joint_data, ax=axes[0,0])
            axes[0,0].set_title('Mean Angular Velocity (Traj1)')
            axes[0,0].set_ylabel('Velocity (degrees/s)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(x='participant', y='Mean_Velocity_Traj2', data=joint_data, ax=axes[0,1])
            axes[0,1].set_title('Mean Angular Velocity (Traj2)')
            axes[0,1].set_ylabel('Velocity (degrees/s)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Acceleration
            sns.boxplot(x='participant', y='Mean_Acceleration_Traj1', data=joint_data, ax=axes[1,0])
            axes[1,0].set_title('Mean Angular Acceleration (Traj1)')
            axes[1,0].set_ylabel('Acceleration (degrees/s²)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(x='participant', y='Mean_Acceleration_Traj2', data=joint_data, ax=axes[1,1])
            axes[1,1].set_title('Mean Angular Acceleration (Traj2)')
            axes[1,1].set_ylabel('Acceleration (degrees/s²)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Jerk
            sns.boxplot(x='participant', y='Mean_Jerk_Traj1', data=joint_data, ax=axes[2,0])
            axes[2,0].set_title('Mean Angular Jerk (Traj1)')
            axes[2,0].set_ylabel('Jerk (degrees/s³)')
            axes[2,0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(x='participant', y='Mean_Jerk_Traj2', data=joint_data, ax=axes[2,1])
            axes[2,1].set_title('Mean Angular Jerk (Traj2)')
            axes[2,1].set_ylabel('Jerk (degrees/s³)')
            axes[2,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ang_{joint}_kinematic_metrics.png'))
            plt.close()
            
            # 3. Advanced Metrics
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Advanced Metrics by Participant - {joint}', y=1.02)
            
            # Error Entropy
            sns.boxplot(x='participant', y='Error_Entropy', data=joint_data, ax=axes[0])
            axes[0].set_title('Error Entropy')
            axes[0].set_ylabel('Entropy (bits)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Error Autocorrelation
            sns.boxplot(x='participant', y='Error_Autocorrelation', data=joint_data, ax=axes[1])
            axes[1].set_title('Error Autocorrelation')
            axes[1].set_ylabel('Correlation')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ang_{joint}_advanced_metrics.png'))
            plt.close()

def generate_statistics(ee_data, ang_data):
    """Generate statistical analysis using the actual Mean_Error and Std_Error values."""
    logger.info("\nGenerating statistics")
    stats = {}
    
    if not ee_data.empty:
        logger.info("Processing end-effector statistics")
        stats['end_effector'] = {
            'mean_error': ee_data['Mean_Error'].mean(),
            'std_error': ee_data['Std_Error'].mean(),
            'by_participant': {
                'mean_error': ee_data.groupby('participant')['Mean_Error'].mean().to_dict(),
                'std_error': ee_data.groupby('participant')['Std_Error'].mean().to_dict()
            },
            'by_trial': {
                'mean_error': ee_data.groupby('trial')['Mean_Error'].mean().to_dict(),
                'std_error': ee_data.groupby('trial')['Std_Error'].mean().to_dict()
            }
        }
    
    if not ang_data.empty:
        logger.info("Processing angular statistics")
        stats['angular'] = {
            'by_joint': {
                joint: {
                    'mean_error': ang_data[ang_data['Joint'] == joint]['Mean_Error'].mean(),
                    'std_error': ang_data[ang_data['Joint'] == joint]['Std_Error'].mean()
                } for joint in ang_data['Joint'].unique()
            },
            'by_participant': {
                participant: {
                    joint: {
                        'mean_error': ang_data[(ang_data['participant'] == participant) & 
                                             (ang_data['Joint'] == joint)]['Mean_Error'].mean(),
                        'std_error': ang_data[(ang_data['participant'] == participant) & 
                                            (ang_data['Joint'] == joint)]['Std_Error'].mean()
                    } for joint in ang_data['Joint'].unique()
                } for participant in ang_data['participant'].unique()
            },
            'by_trial': {
                trial: {
                    joint: {
                        'mean_error': ang_data[(ang_data['trial'] == trial) & 
                                             (ang_data['Joint'] == joint)]['Mean_Error'].mean(),
                        'std_error': ang_data[(ang_data['trial'] == trial) & 
                                            (ang_data['Joint'] == joint)]['Std_Error'].mean()
                    } for joint in ang_data['Joint'].unique()
                } for trial in ang_data['trial'].unique()
            }
        }
    
    return stats

def main():
    # Get the absolute path to the workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    base_path = workspace_root / 'src' / 'manipulator_skill_acquisition' / 'resources' / 'dados_excel'
    output_dir = workspace_root / 'src' / 'manipulator_skill_acquisition' / 'analysis' / 'data' / 'visualizations'
    
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    ee_data, ang_data = load_all_data(base_path)
    
    # Generate visualizations
    generate_visualizations(ee_data, ang_data, output_dir)
    
    # Generate statistics
    stats = generate_statistics(ee_data, ang_data)
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("MANIPULATOR SKILL ACQUISITION - DATA SUMMARY")
    logger.info("="*50)
    
    if 'end_effector' in stats:
        logger.info("\nPOSITION CONTROL PERFORMANCE")
        logger.info("-"*50)
        logger.info(f"Overall Position Accuracy: {stats['end_effector']['mean_error']:.4f} mm")
        logger.info(f"Overall Position Consistency: {stats['end_effector']['std_error']:.4f} mm")
        
        logger.info("\nParticipant Performance (Position Control):")
        logger.info("-"*50)
        for participant in stats['end_effector']['by_participant']['mean_error']:
            mean_error = stats['end_effector']['by_participant']['mean_error'][participant]
            std_error = stats['end_effector']['by_participant']['std_error'][participant]
            logger.info(f"\n{participant.title()}:")
            logger.info(f"  Average Distance from Target: {mean_error:.4f} mm")
            logger.info(f"  Position Consistency: {std_error:.4f} mm")
    
    if 'angular' in stats:
        logger.info("\nJOINT CONTROL PERFORMANCE")
        logger.info("-"*50)
        
        # Overall joint performance
        logger.info("\nOverall Joint Performance:")
        logger.info("-"*30)
        for joint in stats['angular']['by_joint']:
            mean_error = stats['angular']['by_joint'][joint]['mean_error']
            std_error = stats['angular']['by_joint'][joint]['std_error']
            logger.info(f"\nJoint {joint}:")
            logger.info(f"  Average Angle Error: {mean_error:.4f} degrees")
            logger.info(f"  Angle Consistency: {std_error:.4f} degrees")
        
        # Participant-specific joint performance
        logger.info("\nParticipant Performance (Joint Control):")
        logger.info("-"*50)
        for participant in stats['angular']['by_participant']:
            logger.info(f"\n{participant.title()}:")
            for joint in stats['angular']['by_participant'][participant]:
                mean_error = stats['angular']['by_participant'][participant][joint]['mean_error']
                std_error = stats['angular']['by_participant'][participant][joint]['std_error']
                logger.info(f"  Joint {joint}:")
                logger.info(f"    Average Angle Error: {mean_error:.4f} degrees")
                logger.info(f"    Angle Consistency: {std_error:.4f} degrees")
    
    logger.info("\n" + "="*50)
    logger.info("Visualizations have been saved to the 'visualizations' directory")
    logger.info("="*50)

if __name__ == "__main__":
    main() 