import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch
import math


# SAC Networks
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr=0.0001):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 256)
        self.ln_1 = nn.LayerNorm(256)
        self.fc_2 = nn.Linear(256, 256)
        self.ln_2 = nn.LayerNorm(256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)

        # Improved weight initialization
        self._init_weights()

        self.lr = actor_lr

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 1
        self.max_action = 2.0
        self.min_action = -2.0
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _init_weights(self):
        # Initialize weights using He initialization for better training dynamics
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        
        # Initialize output layers with smaller weights
        nn.init.uniform_(self.fc_mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_mu.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_std.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = F.leaky_relu(self.ln_1(self.fc_1(x)))
        x = F.leaky_relu(self.ln_2(self.fc_2(x)))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    @torch.jit.export
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # Enforcing Action Bound with numerically stable calculations
        log_prob = reparameter.log_prob(x_t)
        # More numerically stable version
        log_prob = log_prob - torch.sum(
            torch.log(self.action_scale * (1 - y_t.pow(2) + 1e-6)), 
            dim=-1, 
            keepdim=True
        )

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr=0.0001):
        super(QNetwork, self).__init__()

        self.fc_s = nn.Linear(state_dim, 128)
        self.ln_s = nn.LayerNorm(128)
        self.fc_a = nn.Linear(action_dim, 128)
        self.ln_a = nn.LayerNorm(128)
        self.fc_1 = nn.Linear(256, 256)
        self.ln_1 = nn.LayerNorm(256)
        self.fc_out = nn.Linear(256, action_dim)

        # Improved weight initialization
        self._init_weights()

        self.lr = critic_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def _init_weights(self):
        # Initialize weights using He initialization for better training dynamics
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        
        # Initialize output layer with smaller weights
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.ln_s(self.fc_s(x)))
        h2 = F.leaky_relu(self.ln_a(self.fc_a(a)))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.leaky_relu(self.ln_1(self.fc_1(cat)))
        q = self.fc_out(q)
        return q


# TD3/DDPG Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr=0.0001, max_action=2.0):
        super(Actor, self).__init__()
        
        self.fc_1 = nn.Linear(state_dim, 256)
        self.ln_1 = nn.LayerNorm(256)
        self.fc_2 = nn.Linear(256, 256)
        self.ln_2 = nn.LayerNorm(256)
        self.fc_out = nn.Linear(256, action_dim)
        
        # TD3 uses deterministic policy, so we don't need std output
        self.max_action = max_action
        
        # Improved weight initialization
        self._init_weights()
        
        self.lr = actor_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def _init_weights(self):
        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        
        # Initialize output layer with smaller weights for stability
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)
    
    def forward(self, x):
        x = F.leaky_relu(self.ln_1(self.fc_1(x)))
        x = F.leaky_relu(self.ln_2(self.fc_2(x)))
        # TD3 uses tanh activation for actions
        action = self.max_action * torch.tanh(self.fc_out(x))
        return action
    
    def get_action(self, state, add_noise=False, noise_scale=0.1):
        with torch.no_grad():
            action = self.forward(state)
            if add_noise:
                # Add clipped normal noise for exploration
                noise = torch.randn_like(action) * noise_scale
                noise = torch.clamp(noise, -0.5, 0.5)
                action = action + noise
                action = torch.clamp(action, -self.max_action, self.max_action)
            return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr=0.0001):
        super(Critic, self).__init__()
        
        # Q1 architecture - first critic network
        self.q1_fc_s = nn.Linear(state_dim, 256)
        self.q1_ln_s = nn.LayerNorm(256)
        self.q1_fc_a = nn.Linear(action_dim, 256)
        self.q1_ln_a = nn.LayerNorm(256)
        self.q1_fc_1 = nn.Linear(512, 256)
        self.q1_ln_1 = nn.LayerNorm(256)
        self.q1_fc_out = nn.Linear(256, 1)
        
        # Q2 architecture - second critic network (for TD3's twin critics)
        self.q2_fc_s = nn.Linear(state_dim, 256)
        self.q2_ln_s = nn.LayerNorm(256)
        self.q2_fc_a = nn.Linear(action_dim, 256)
        self.q2_ln_a = nn.LayerNorm(256)
        self.q2_fc_1 = nn.Linear(512, 256)
        self.q2_ln_1 = nn.LayerNorm(256)
        self.q2_fc_out = nn.Linear(256, 1)
        
        # Improved weight initialization
        self._init_weights()
        
        self.lr = critic_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
    
    def _init_weights(self):
        # Initialize weights for both Q networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        
        # Initialize output layers with smaller weights
        for fc_out in [self.q1_fc_out, self.q2_fc_out]:
            nn.init.uniform_(fc_out.weight, -3e-3, 3e-3)
            nn.init.uniform_(fc_out.bias, -3e-3, 3e-3)
    
    def forward_q1(self, state, action):
        # First critic forward pass
        s = F.leaky_relu(self.q1_ln_s(self.q1_fc_s(state)))
        a = F.leaky_relu(self.q1_ln_a(self.q1_fc_a(action)))
        q1 = torch.cat([s, a], dim=-1)
        q1 = F.leaky_relu(self.q1_ln_1(self.q1_fc_1(q1)))
        q1 = self.q1_fc_out(q1)
        return q1
    
    def forward_q2(self, state, action):
        # Second critic forward pass
        s = F.leaky_relu(self.q2_ln_s(self.q2_fc_s(state)))
        a = F.leaky_relu(self.q2_ln_a(self.q2_fc_a(action)))
        q2 = torch.cat([s, a], dim=-1)
        q2 = F.leaky_relu(self.q2_ln_1(self.q2_fc_1(q2)))
        q2 = self.q2_fc_out(q2)
        return q2
    
    def forward(self, state, action):
        # Return both Q-values
        q1 = self.forward_q1(state, action)
        q2 = self.forward_q2(state, action)
        return q1, q2