from typing import Dict, List
import numpy as np

class ConfigurationSpaceManager:
    def __init__(self, config_space: Dict[str, List], transitions: Dict[str, Dict], id_to_type: Dict[str, str] = None):
        self.config_space = config_space
        self.transitions = transitions
        self.entity_states = {}
        self.id_to_type = id_to_type or {}

    def _get_entity_type(self, entity_id: str) -> str:
        return self.id_to_type.get(entity_id, entity_id)

    def get_valid_configs(self, entity_id: str) -> List:
        return self.config_space.get(self._get_entity_type(entity_id), [])

    def get_current_config(self, entity_id: str) -> str:
        return self.entity_states.get(entity_id, None)

    def update_state(self, entity_id: str, new_config: str):
        if new_config in self.get_valid_configs(entity_id):
            self.entity_states[entity_id] = new_config
        else:
            raise ValueError(f"Invalid configuration {new_config}")

    def step(self, entity_id: str) -> str:
        current = self.get_current_config(entity_id)
        if current is None:
            raise RuntimeError("Entity has no initialized state.")

        next_configs = self.transitions.get(current, {})
        if not next_configs:
            return current

        choices = list(next_configs.keys())
        probs = list(next_configs.values())
        next_config = np.random.choice(choices, p=probs)
        self.update_state(entity_id, next_config)
        return next_config

    def build_distribution(self) -> np.ndarray:
        all_configs = []
        for configs in self.config_space.values():
            all_configs.extend(configs)
        
        indices = {config: idx for idx, config in enumerate(all_configs)}
        dist = np.zeros(len(all_configs))
        
        for config in self.entity_states.values():
            if config in indices:
                dist[indices[config]] += 1
                
        return dist / len(self.entity_states)
    
    def validate_initialization(self, entity_ids: List[str]):
        missing = []
        for eid in entity_ids:
            valid_configs = self.get_valid_configs(eid)
            if not valid_configs:
                missing.append(eid)
        
        if missing:
            raise ValueError(f"No valid configurations found for entities: {missing}")