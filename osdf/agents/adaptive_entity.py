from typing import List, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
from osdf.core.configuration_manager import ConfigurationSpaceManager

class NeuralNetworkAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def train_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class AdaptiveEntity:
    def __init__(self, entity_id: str, config_space: List[str], input_dim: int = None):
        self.id = entity_id
        self.config_space = config_space
        self.model = NeuralNetworkAdapter(
            input_dim or len(config_space), 
            len(config_space)
        )
        self.current_config_idx = 0
        
    def get_input_vector(self, context: Dict) -> torch.Tensor:
        input_vec = torch.zeros(len(self.config_space))
        input_vec[self.current_config_idx] = 1.0
        return input_vec.unsqueeze(0)
        
    def predict_configuration(self, context: Dict) -> str:
        with torch.no_grad():
            inputs = self.get_input_vector(context)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            _, predicted_idx = torch.max(probs, 1)
            self.current_config_idx = predicted_idx.item()
            return self.config_space[self.current_config_idx]
        
    def learn(self, context: Dict, reward: float):
        inputs = self.get_input_vector(context)
        logits = self.model(inputs)
        probs = torch.softmax(logits, dim=1)
        
        action_log_prob = torch.log(probs[0, self.current_config_idx])
        loss = -action_log_prob * reward
        
        self.model.train_step(loss)