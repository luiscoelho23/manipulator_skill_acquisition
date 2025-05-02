import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import env_dmp_obstacle as Env
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import yaml
import sys
import signal
import atexit
from datetime import datetime
from nn import Actor, Critic  # Import the TD3-specific network classes

# Flag to track if training should continue
training_active = True

# Track if we're currently in an episode
in_episode = False

# Reference to current agent for saving
current_agent = None
current_output_config = None

# Function to save model on exit
def save_model_on_exit():
    global current_agent, current_output_config, training_active
    if not training_active or current_agent is None or current_output_config is None:
        return
        
    try:
        print("Saving model before exit...")
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(current_output_config['directory'], 
                           f"td3_{current_output_config['model_name']}_{current_date}_interrupted.pt")
        torch.save(current_agent.actor.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Signal handler for graceful termination
def handle_signal(signum, frame):
    global training_active, in_episode
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\nReceived signal {signal_name}. Stopping training gracefully...")
    training_active = False
    
    # If we're not in an episode, exit immediately
    if not in_episode:
        save_model_on_exit()
        sys.exit(0)

# Register exit handler
atexit.register(save_model_on_exit)

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)
try:
    signal.signal(signal.SIGBREAK, handle_signal)  # Windows-specific
except AttributeError:
    pass  # Not on Windows

class ReplayBuffer:
    def __init__(self, buffer_limit, state_dim, action_dim, device):
        self.buffer_limit = int(buffer_limit)
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Pre-allocate memory for better performance
        self.states = np.zeros((self.buffer_limit, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_limit, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_limit, 1), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_limit, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_limit, 1), dtype=np.float32)
        
        # For prioritized experience replay
        self.use_per = False
        self.priorities = np.zeros((self.buffer_limit,), dtype=np.float32)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to prevent zero priority
        
    def put(self, transition):
        s, a, r, s_prime, done = transition
        
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_prime
        self.dones[self.ptr] = 0.0 if done else 1.0
        
        # Initial high priority for new experiences
        self.priorities[self.ptr] = np.max(self.priorities) if self.size > 0 else 1.0
        
        self.ptr = (self.ptr + 1) % self.buffer_limit
        self.size = min(self.size + 1, self.buffer_limit)
        
    def sample(self, n):
        if self.use_per:
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Compute sampling probabilities
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            
            # Sample indices based on prioritized probabilities
            indices = np.random.choice(self.size, n, replace=False, p=probs)
            
            # Compute importance sampling weights
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        else:
            indices = np.random.choice(self.size, min(n, self.size), replace=False)
            weights = None
            
        # Get batch data
        s_batch = torch.FloatTensor(self.states[indices]).to(self.device)
        a_batch = torch.FloatTensor(self.actions[indices]).to(self.device)
        r_batch = torch.FloatTensor(self.rewards[indices]).to(self.device)
        s_prime_batch = torch.FloatTensor(self.next_states[indices]).to(self.device)
        done_batch = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        # Normalize rewards for better stability
        if len(r_batch) > 1:  # Only normalize if we have enough samples
            r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)
            
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        if self.use_per:
            priorities = np.abs(priorities.detach().cpu().numpy()) + self.epsilon
            self.priorities[indices] = priorities
    
    def enable_per(self, enable=True):
        """Enable or disable Prioritized Experience Replay"""
        self.use_per = enable
        
    def __len__(self):
        return self.size

class TD3_Agent:
    def __init__(self, state_dim=7, action_dim=3, learning_rate=0.0001, gamma=0.99, 
                 tau=0.005, batch_size=256, buffer_limit=1000000, exploration_noise=0.1,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 target_update_interval=1, gradient_steps=1):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = learning_rate
        self.lr_critic = learning_rate * 1.25
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_limit = int(buffer_limit)
        self.tau = tau
        
        # TD3-specific parameters
        self.exploration_noise = exploration_noise  # Noise for exploration
        self.policy_noise = policy_noise            # Noise added to target actions
        self.noise_clip = noise_clip                # Clipping of target policy noise
        self.policy_delay = policy_delay            # Delayed policy updates frequency
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.buffer_limit, state_dim, action_dim, self.DEVICE)
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.training_step = 0
        self.max_action = 2.0  # Action limit
        
        # Initialize Actor
        self.actor = Actor(self.state_dim, self.action_dim, self.lr_actor, self.max_action).to(self.DEVICE)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.lr_actor, self.max_action).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize Critic (Twin critics)
        self.critic = Critic(self.state_dim, self.action_dim, self.lr_critic).to(self.DEVICE)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.lr_critic).to(self.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Disable gradient calculations for target networks
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def choose_action(self, s, eval_mode=False):
        with torch.no_grad():
            state = s.to(self.DEVICE)
            if eval_mode:
                # For evaluation, use deterministic policy without noise
                action = self.actor(state)
            else:
                # During training, add exploration noise
                action = self.actor(state)
                noise = torch.randn_like(action) * self.exploration_noise
                noise = torch.clamp(noise, -0.5, 0.5)
                action = action + noise
                action = torch.clamp(action, -self.max_action, self.max_action)
            
            return action.detach().cpu().numpy()
    
    def train_agent(self):
        self.training_step += 1
        
        for _ in range(self.gradient_steps):
            if len(self.memory) < self.batch_size:
                return
                
            mini_batch = self.memory.sample(self.batch_size)
            s_batch, a_batch, r_batch, s_prime_batch, done_batch, indices, weights = mini_batch
            
            # Compute critic loss - using twin critics
            with torch.no_grad():
                # Select action from target policy with noise for smoothing
                noise = torch.randn_like(a_batch) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                
                next_actions = self.actor_target(s_prime_batch) + noise
                next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
                
                # Get target Q values from both critics
                target_q1, target_q2 = self.critic_target(s_prime_batch, next_actions)
                
                # Take the minimum of the two target Q-values to reduce overestimation
                target_q = torch.min(target_q1, target_q2)
                target = r_batch + self.gamma * done_batch * target_q
            
            # Get current Q estimates from both critics
            current_q1, current_q2 = self.critic(s_batch, a_batch)
            
            # Calculate TD errors for priority updates if using PER
            with torch.no_grad():
                td_errors1 = target - current_q1
                td_errors2 = target - current_q2
                td_errors = (td_errors1.abs() + td_errors2.abs()) / 2.0
            
            # Compute critic loss for both critics
            critic_loss1 = F.smooth_l1_loss(current_q1, target, reduction='none')
            critic_loss2 = F.smooth_l1_loss(current_q2, target, reduction='none')
            
            if weights is not None:
                critic_loss1 = (critic_loss1 * weights).mean()
                critic_loss2 = (critic_loss2 * weights).mean()
            else:
                critic_loss1 = critic_loss1.mean()
                critic_loss2 = critic_loss2.mean()
            
            critic_loss = critic_loss1 + critic_loss2
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic.optimizer.step()
            
            # Delayed policy updates
            if self.training_step % self.policy_delay == 0:
                # Compute actor loss (policy gradient)
                actor_loss = -self.critic.forward_q1(s_batch, self.actor(s_batch)).mean()
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor.optimizer.step()
                
                # Update target networks with soft update
                for param_target, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
                
                for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
            
            # Update priorities in replay buffer if using PER
            if self.memory.use_per:
                self.memory.update_priorities(indices, td_errors)

def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path):
    """Main training function that accepts configuration file path."""
    global current_agent, current_output_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters from config
    env_name = config['environment']
    hyperparams = config['hyperparameters']
    dmp_config = config['dmp']
    output_config = config['output']
    
    # Store output config for exit handler
    current_output_config = output_config
    
    # Create output directory if it doesn't exist
    os.makedirs(output_config['directory'], exist_ok=True)
    
    # Initialize environment
    try:
        if env_name == "env_dmp_obstacle":
            dmp_file = dmp_config.get('file', '')
            print(f"Using DMP file: {dmp_file}")
            env = Env.DmpObstacleEnv(dmp_path=dmp_file)
        elif env_name == "env_dmp_obstacle_via_points":
            dmp_file = dmp_config.get('file', '')
            print(f"Using DMP file: {dmp_file}")
            env = Env.DmpObstacleViaPointsEnv(dmp_path=dmp_file)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    except TypeError as e:
        print(f"Error initializing environment: {e}")
        print("Trying alternative initialization...")
        if env_name == "env_dmp_obstacle":
            env = Env.DmpObstacleEnv()
        elif env_name == "env_dmp_obstacle_via_points":
            env = Env.DmpObstacleViaPointsEnv()
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    
    # Verify environment has observation and action spaces
    if not hasattr(env, 'observation_space'):
        # Create a simple space for observations
        print("Environment does not have observation_space, creating a default one.")
        state_dim = 7  # Default based on dimensions in other implementations
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
    
    if not hasattr(env, 'action_space'):
        # Create a simple space for actions
        print("Environment does not have action_space, creating a default one.")
        action_dim = 3  # Default based on dimensions in other implementations
        env.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(action_dim,))
    
    # Determine whether to use CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for training")
    
    # Extract TD3-specific parameters with defaults
    policy_noise = float(hyperparams.get('policy_noise', 0.2))
    noise_clip = float(hyperparams.get('noise_clip', 0.5))
    policy_delay = int(hyperparams.get('policy_delay', 2))
    target_update_interval = int(hyperparams.get('target_update_interval', 1))
    gradient_steps = int(hyperparams.get('gradient_steps', 1))
    use_per = hyperparams.get('use_prioritized_replay', False)
    noise_decay = hyperparams.get('noise_decay', 0.995)
    min_noise = hyperparams.get('min_noise', 0.01)
    
    # Initialize agent with hyperparameters
    agent = TD3_Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=float(hyperparams.get('learning_rate', 0.0001)),
        gamma=float(hyperparams.get('gamma', 0.99)),
        tau=float(hyperparams.get('tau', 0.005)),
        batch_size=int(hyperparams.get('batch_size', 256)),
        buffer_limit=int(hyperparams.get('buffer_limit', 1000000)),
        exploration_noise=float(hyperparams.get('exploration_noise', 0.1)),
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        target_update_interval=target_update_interval,
        gradient_steps=gradient_steps
    )
    
    # Enable prioritized experience replay if configured
    if use_per:
        agent.memory.enable_per(True)
        print("Using prioritized experience replay")
    
    # Store agent for exit handler
    current_agent = agent
    
    # Training loop
    EPISODE = 10000
    early_stopping_patience = hyperparams.get('early_stopping_patience', 200)
    early_stopping_threshold = hyperparams.get('early_stopping_threshold', 0.01)
    use_evaluation = hyperparams.get('use_evaluation', False)
    eval_interval = hyperparams.get('eval_interval', 10)
    best_eval_score = -float('inf')
    no_improvement_counter = 0
    
    print_once = True
    score_list = []
    eval_score_list = []
    train_ep = 0
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_noise = agent.exploration_noise
    
    for EP in range(EPISODE):
        # Check if we should stop training
        if not training_active:
            print("Training stopped by user.")
            break
            
        state, _ = env.reset()
        score, done = 0.0, False
        
        # We're now entering an episode
        global in_episode
        in_episode = True
        
        # Reduce exploration noise over time
        current_noise = max(min_noise, current_noise * noise_decay)
        agent.exploration_noise = current_noise
        
        while not done:
            action = agent.choose_action(torch.FloatTensor(state))
            
            try:
                state_prime, reward, done, info, _ = env.step(action)
            except ValueError:
                # Handle case where env.step() returns fewer values (e.g., older Gym API)
                try:
                    state_prime, reward, done, info = env.step(action)
                    _ = {}
                except ValueError as e:
                    print(f"Error in env.step(): {e}")
                    print("Please check the environment's step method signature.")
                    sys.exit(1)
            
            agent.memory.put((state, action, reward, state_prime, done))
            
            score += reward
            
            state = state_prime
            
            # Check if we should stop training during an episode
            if not training_active:
                done = True
                break
            
            if len(agent.memory) > 10000:
                if not train_ep:
                    print("Starting training...")
                    train_ep = EP
                
                agent.train_agent()
                
        # We've finished an episode
        in_episode = False
        
        # If training was stopped during this episode, exit now
        if not training_active:
            break
        
        print(f"EP:{EP}, Score:{score:.1f}, Noise:{current_noise:.4f}, Policy updates:{agent.training_step // agent.policy_delay}")
        score_list.append(score)
        
        # Run evaluation if configured
        if use_evaluation and EP % eval_interval == 0:
            eval_score = evaluate(env, agent, 5)  # Run 5 evaluation episodes
            eval_score_list.append(eval_score)
            print(f"Evaluation at EP:{EP}, Score:{eval_score:.1f}")
            
            # Check if this is the best model
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                no_improvement_counter = 0
                
                # Save best model
                if output_config.get('save_best', True):
                    model_path = os.path.join(output_config['directory'], 
                                            f"td3_{output_config['model_name']}_{current_date}_best.pt")
                    torch.save(agent.actor.state_dict(), model_path)
            else:
                no_improvement_counter += 1
        else:
            # If not using evaluation, save based on training score
            if output_config.get('save_best', True) and score > max(score_list[:-1] or [float('-inf')]):
                model_path = os.path.join(output_config['directory'], 
                                        f"td3_{output_config['model_name']}_{current_date}_best.pt")
                torch.save(agent.actor.state_dict(), model_path)
        
        # Save model according to configuration
        if output_config.get('save_checkpoints', True) and EP % output_config.get('save_frequency', 100) == 0:
            model_path = os.path.join(output_config['directory'], 
                                    f"td3_{output_config['model_name']}_{current_date}_EP{EP}.pt")
            torch.save(agent.actor.state_dict(), model_path)
        
        # Plot scores
        if EP % 50 == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(score_list[train_ep:])
            plt.title('Training Scores')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            if use_evaluation and eval_score_list:
                plt.subplot(1, 2, 2)
                plt.plot(eval_score_list)
                plt.title('Evaluation Scores')
                plt.xlabel('Evaluation')
                plt.ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_config['directory'], 'training_progress.png'))
            plt.close()
        
        # Early stopping
        if no_improvement_counter >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} evaluations. Stopping training.")
            break
            
        # Also check for convergence in training scores
        if EP > 100:  # Need some history to check convergence
            recent_scores = score_list[-20:]  # Last 20 episodes
            if np.std(recent_scores) < early_stopping_threshold * np.abs(np.mean(recent_scores)):
                print("Training scores have converged. Stopping training.")
                break

def evaluate(env, agent, episodes=5):
    """Run evaluation episodes without exploration"""
    eval_scores = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            # Use deterministic action selection for evaluation
            action = agent.choose_action(torch.FloatTensor(state), eval_mode=True)
            
            try:
                next_state, reward, done, _, _ = env.step(action)
            except ValueError:
                next_state, reward, done, _ = env.step(action)
                
            score += reward
            state = next_state
            
        eval_scores.append(score)
        
    return np.mean(eval_scores)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_td3.py <config_path>")
        sys.exit(1)
    
    train(sys.argv[1]) 