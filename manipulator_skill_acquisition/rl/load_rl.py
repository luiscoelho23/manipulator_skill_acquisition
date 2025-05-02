import torch
import numpy as np
import os
from .nn import PolicyNetwork, Actor

class RLActor:
    def __init__(self, actor_path, model_type=None):
        # Basic setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 7
        
        # Detect model type (inline)
        if model_type is None:
            model_type = 'sac' if 'sac' in os.path.basename(actor_path).lower() else 'ddpg'
        self.model_type = model_type.lower()
        self.is_sac = self.model_type == 'sac'
        
        # Load dimensions from state dict
        try:
            state_dict = torch.load(actor_path, map_location='cpu', weights_only=True)
            self.output_dim = state_dict.get('fc_mu.weight', state_dict.get('fc_out.weight', None))
            self.output_dim = self.output_dim.shape[0] if self.output_dim is not None else 3
            self.input_dim = state_dict.get('fc_1.weight', None)
            self.input_dim = self.input_dim.shape[1] if self.input_dim is not None else 7
        except:
            self.output_dim = 3
            
        self.max_action = 2.0
        
        # Load model with minimal overhead
        try:
            if self.is_sac:
                self.actor = PolicyNetwork(self.input_dim, self.output_dim).to(self.device)
            else:
                self.actor = Actor(self.input_dim, self.output_dim, max_action=self.max_action).to(self.device)
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device, weights_only=True), strict=False)
        except:
            if self.is_sac:
                self.actor = PolicyNetwork(self.input_dim, self.output_dim).to(self.device)
            else:
                self.actor = Actor(self.input_dim, self.output_dim, max_action=self.max_action).to(self.device)
        
        # Set model to evaluation mode and disable gradients globally
        self.actor.eval()
        torch.set_grad_enabled(False)
        
        # Pre-allocate fixed tensors assuming 7-element input (common case)
        self.input_tensor = torch.zeros(1, 7, device=self.device)
        
        # Pre-extract direct model methods
        if self.is_sac:
            self._forward = self.actor.sample
        else:
            self._forward = self.actor
    
    @torch.no_grad()
    def get_action(self, input_data):
        # Bare minimum direct tensor assignment (no bounds checking)
        self.input_tensor[0, 0] = input_data[0]
        self.input_tensor[0, 1] = input_data[1]
        self.input_tensor[0, 2] = input_data[2]
        self.input_tensor[0, 3] = input_data[3]
        self.input_tensor[0, 4] = input_data[4]
        self.input_tensor[0, 5] = input_data[5]
        self.input_tensor[0, 6] = input_data[6]
        
        # Direct inference
        if self.is_sac:
            action, _ = self._forward(self.input_tensor)
        else:
            action = self._forward(self.input_tensor)
        
        # Fast output
        return action[0].cpu().numpy()