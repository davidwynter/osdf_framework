from typing import Dict


class EntityInteractionEngine:
    def __init__(
        self,
        config_manager: 'ConfigurationSpaceManager',
        graph_modeler: 'EnvironmentGraphModeler',
        signal_propagator: 'ObjectiveSignalPropagator'
    ):
        """
        Coordinates local interactions between entities.
        
        Parameters:
        - config_manager: Manages entity configurations
        - graph_modeler: Defines neighborhood relationships
        - signal_propagator: Handles objective signal exchange
        """
        self.config_mgr = config_manager
        self.graph_model = graph_modeler
        self.signal_prop = signal_propagator

    def synchronize_states(self):
        """For each entity, update state based on neighbors and objective signals."""
        updated_states = {}

        for entity_id in self.graph_model.graph.nodes:
            try:
                next_config = self.config_mgr.step(entity_id)
                updated_states[entity_id] = next_config
            except Exception as e:
                print(f"Error updating {entity_id}: {str(e)}")

        self.config_mgr.entity_states.update(updated_states)

    def propagate_and_update(self):
        """Step-by-step interaction logic."""
        # 1. Propagate objective signals
        self.signal_prop.propagate_signals()

        # 2. Synchronize entity states
        self.synchronize_states()

        # 3. Update stationary distribution
        new_dist = self.config_mgr.build_distribution()
        self.signal_prop.update_stationary_distribution(new_dist)

    def build_entity_action_map(self) -> Dict[str, str]:
        """Return mapping of entity → current configuration for logging."""
        return {
            eid: self.config_mgr.get_current_config(eid)
            for eid in self.graph_model.graph.nodes
        }