from typing import Dict, List

class Entity:
    def __init__(self, entity_id: str, config_space: List[str]):
        self.id = entity_id
        self.config_space = config_space
        self.current_config_idx = 0
        
    def get_input_vector(self, context: Dict) -> Dict:
        input_data = {
            "current_config": self.config_space[self.current_config_idx],
            "neighbors": [],
            "signals": {}
        }
        return input_data
        
    def predict_configuration(self, context: Dict) -> str:
        neighbors = context.get("neighbors", [])
        signals = context.get("signals", {})
        
        if neighbors and signals:
            strongest_signal = max(signals.items(), key=lambda x: x[1])
            return strongest_signal[0]
            
        return self.config_space[self.current_config_idx]