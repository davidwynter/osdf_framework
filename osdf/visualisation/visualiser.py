import matplotlib.pyplot as plt

class Visualiser:
    def plot_action_trajectories(self, log_df, entity_ids=None):
        plt.figure(figsize=(12, 6))
        
        if entity_ids is None:
            entity_ids = [col.replace('action_', '') for col in log_df.columns if col.startswith('action_')]
        
        for entity_id in entity_ids:
            action_col = f'action_{entity_id}'
            if action_col in log_df.columns:
                unique_actions = log_df[action_col].unique()
                action_map = {act: i for i, act in enumerate(unique_actions)}
                trajectory = log_df[action_col].map(action_map)
                
                plt.plot(log_df['timestep'], trajectory, label=f'Entity {entity_id}', marker='o')
        
        plt.xlabel('Timestep')
        plt.ylabel('Action Index')
        plt.title('Entity Action Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()