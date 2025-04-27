import torch
import numpy as np
from .nn import PolicyNetwork

class RLActor:
    def __init__(self, actor_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 7  
        self.output_dim = 5
        self.actor = self._load_actor(actor_path)
    
    def _load_actor(self, actor_path):
        actor = PolicyNetwork(self.input_dim, self.output_dim).to(self.device)
        actor.load_state_dict(torch.load(actor_path, weights_only=True), strict=False)
        actor.eval()  # Coloca o modelo em modo de avaliação
        return actor
    
    def _get_action_values(self, state):
        with torch.no_grad():
            action, log_prob = self.actor.sample(state.to(self.device))
        return action.cpu().numpy()
    
    def get_action(self, input_data):
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        action = self._get_action_values(input_tensor)

        return action
