import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import env_dmp_obstacle as Env
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from nn import PolicyNetwork, QNetwork
import yaml
import sys
import signal
import atexit
from datetime import datetime

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
                           f"sac_{current_output_config['model_name']}_{current_date}_interrupted.pt")
        torch.save(current_agent.PI.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Signal handler for graceful termination
def handle_signal(signum, frame):
    global training_active, in_episode
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\nReceived signal {signal_name}. Stopping training gracefully...")
    training_active = False
    
    # If we're in an episode, let it finish
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


class SAC_Agent:
    def __init__(self, state_dim=7, action_dim=3, learning_rate=0.0003, gamma=0.99, 
                 tau=0.01, batch_size=256, init_alpha=0.1, buffer_limit=1000000,
                 target_update_interval=1, gradient_steps=1, use_automatic_entropy_tuning=True):

        self.state_dim = state_dim  
        self.action_dim = action_dim 
        self.lr_pi = learning_rate
        self.lr_q = learning_rate * 1.25
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_limit = int(buffer_limit)
        self.tau = tau
        self.init_alpha = init_alpha
        self.target_entropy = -self.action_dim 
        self.lr_alpha = self.lr_pi/2
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.buffer_limit, state_dim, action_dim, self.DEVICE)
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.training_step = 0

        # Initialize log_alpha and optimizer
        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        else:
            self.alpha = init_alpha

        # Initialize networks
        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        # Initialize target networks with source network params
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        
        # Disable gradient calculations for target networks
        for param in self.Q1_target.parameters():
            param.requires_grad = False
        for param in self.Q2_target.parameters():
            param.requires_grad = False

    def choose_action(self, s, eval_mode=False):
        with torch.no_grad():
            if eval_mode:
                # For evaluation, use mean action without sampling
                mean, _ = self.PI.forward(s.to(self.DEVICE))
                action = torch.tanh(mean) * self.PI.action_scale + self.PI.action_bias
                return action.detach().cpu().numpy(), None
            else:
                # For training, use stochastic policy
                action, log_prob = self.PI.sample(s.to(self.DEVICE))
                return action.detach().cpu().numpy(), log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done, _, _ = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.PI.sample(s_prime)
            if self.use_automatic_entropy_tuning:
                entropy = -self.log_alpha.exp() * log_prob_prime
            else:
                entropy = -self.alpha * log_prob_prime
                
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.gamma * done * (q_target + entropy)
        return target

    def train_agent(self):
        self.training_step += 1
        
        for _ in range(self.gradient_steps):
            if len(self.memory) < self.batch_size:
                return
                
            mini_batch = self.memory.sample(self.batch_size)
            s_batch, a_batch, r_batch, s_prime_batch, done_batch, indices, weights = mini_batch

            td_target = self.calc_target(mini_batch)

            # Calculate TD errors for priority updates if using PER
            with torch.no_grad():
                td_errors1 = td_target - self.Q1(s_batch, a_batch)
                td_errors2 = td_target - self.Q2(s_batch, a_batch)
                td_errors = torch.min(td_errors1.abs(), td_errors2.abs())

            #### Q1 train ####
            q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target, reduction='none')
            if weights is not None:
                q1_loss = (q1_loss * weights).mean()
            else:
                q1_loss = q1_loss.mean()
                
            self.Q1.optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1.0)
            self.Q1.optimizer.step()
            #### Q1 train ####

            #### Q2 train ####
            q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target, reduction='none')
            if weights is not None:
                q2_loss = (q2_loss * weights).mean()
            else:
                q2_loss = q2_loss.mean()
                
            self.Q2.optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1.0)
            self.Q2.optimizer.step()
            #### Q2 train ####

            #### pi train ####
            a, log_prob = self.PI.sample(s_batch)
            if self.use_automatic_entropy_tuning:
                entropy = -self.log_alpha.exp() * log_prob
            else:
                entropy = -self.alpha * log_prob

            q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
            q = torch.min(q1, q2)

            pi_loss = -(q + entropy).mean()  # for gradient ascent
            self.PI.optimizer.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.PI.parameters(), 1.0)
            self.PI.optimizer.step()
            #### pi train ####

            #### alpha train ####
            if self.use_automatic_entropy_tuning:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
            #### alpha train ####

            # Update priorities in replay buffer if using PER
            if self.memory.use_per:
                self.memory.update_priorities(indices, td_errors)

            # Only update target networks periodically
            if self.training_step % self.target_update_interval == 0:
                #### Q1, Q2 soft-update ####
                for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
                for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
                #### Q1, Q2 soft-update ####

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
        state_dim = 7  # Default based on _get_obs method
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
    
    if not hasattr(env, 'action_space'):
        # Create a simple space for actions
        print("Environment does not have action_space, creating a default one.")
        action_dim = 3  # Default based on the policy value
        env.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(action_dim,))
    
    # Determine whether to use CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for training")
    
    # Extract additional parameters with defaults
    target_update_interval = int(hyperparams.get('target_update_interval', 1))
    gradient_steps = int(hyperparams.get('gradient_steps', 1))
    use_per = hyperparams.get('use_prioritized_replay', False)
    use_auto_entropy = hyperparams.get('use_automatic_entropy_tuning', True)
    
    # Initialize agent with hyperparameters
    agent = SAC_Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=float(hyperparams.get('learning_rate', 0.0003)),
        gamma=float(hyperparams.get('gamma', 0.99)),
        tau=float(hyperparams.get('tau', 0.01)),
        batch_size=int(hyperparams.get('batch_size', 256)),
        init_alpha=float(hyperparams.get('init_alpha', 0.1)),
        buffer_limit=int(hyperparams.get('buffer_limit', 1000000)),
        target_update_interval=target_update_interval,
        gradient_steps=gradient_steps,
        use_automatic_entropy_tuning=use_auto_entropy
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
        
        while not done:
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            
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
        
        print("EP:{}, Score:{:.1f}".format(EP, score))
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
                                            f"sac_{output_config['model_name']}_{current_date}_best.pt")
                    torch.save(agent.PI.state_dict(), model_path)
            else:
                no_improvement_counter += 1
        else:
            # If not using evaluation, save based on training score
            if output_config.get('save_best', True) and score > max(score_list[:-1] or [float('-inf')]):
                model_path = os.path.join(output_config['directory'], 
                                        f"sac_{output_config['model_name']}_{current_date}_best.pt")
                torch.save(agent.PI.state_dict(), model_path)
        
        # Save model according to configuration
        if output_config.get('save_checkpoints', True) and EP % output_config.get('save_frequency', 100) == 0:
            model_path = os.path.join(output_config['directory'], 
                                    f"sac_{output_config['model_name']}_{current_date}_EP{EP}.pt")
            torch.save(agent.PI.state_dict(), model_path)
        
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
            action, _ = agent.choose_action(torch.FloatTensor(state), eval_mode=True)
            
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
        print("Usage: python train_sac.py <config_path>")
        sys.exit(1)
    
    train(sys.argv[1]) 