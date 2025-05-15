import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class ObservationLogger:
    def __init__(self):
        self.log_data = []

    def log_timestep(self,
                     t: int,
                     actions: Dict[str, str],
                     objectives: Dict[str, float]):
        """
        Logs a single timestep's data including configurations and objective values.
        
        Parameters:
        - t: Current simulation step number
        - actions: Map of {entity_id: current_config}
        - objectives: Map of {entity_id: current_objective_value}
        """
        entry = {
            "timestep": t,
            "timestamp": datetime.now().isoformat(),
            **{f"action_{eid}": act for eid, act in actions.items()},
            **{f"objective_{eid}": obj for eid, obj in objectives.items()}
        }
        self.log_data.append(entry)

    def export_log(self):
        """
        Returns all logged data as a Pandas DataFrame
        """
        return pd.DataFrame(self.log_data)

    def get_action_history(self, entity_id: str):
        """
        Extracts time-series configuration history for one entity
        """
        action_key = f"action_{entity_id}"
        configs = [entry[action_key] for entry in self.log_data if action_key in entry]
        return list(range(len(configs))), configs

    def visualize_actions(self, entity_ids: List[str], n_steps: Optional[int] = None):
        """
        Visualizes the action trajectories for given entities over time.
        
        Parameters:
        - entity_ids: List of entity IDs to plot
        - n_steps: Number of timesteps to show (default: all available)
        """
        from collections import Counter
        
        # Limit steps if needed
        n_steps = n_steps or len(self.log_data)
        limited_data = self.log_data[:n_steps]

        # Create timeline
        timesteps = list(range(n_steps))
        
        fig, ax = plt.subplots(figsize=(12, 6))

        # For each entity, collect its history
        for entity_id in entity_ids:
            action_key = f"action_{entity_id}"
            
            # Build sequence of actions over time
            actions_over_time = [
                entry.get(action_key, None) for entry in limited_data
            ]
            
            # Convert to numeric indices for plotting
            unique_actions = list(set(actions_over_time))
            action_to_idx = {act: i for i, act in enumerate(unique_actions)}
            
            numeric_actions = [
                action_to_idx[act] for act in actions_over_time 
                if act is not None
            ]

            # Plot as line
            ax.plot(
                timesteps,
                numeric_actions,
                marker="o",
                linestyle="-",
                label=f"Entity {entity_id}"
            )

        # Configure axes and labels
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action State")
        ax.set_title("Entity Configuration Trajectories Over Time")
        ax.legend()
        ax.grid(True)
        
        # Add y-axis tick labels
        ax.set_yticks(range(len(unique_actions)))
        ax.set_yticklabels(unique_actions)
        
        plt.tight_layout()
        plt.show()