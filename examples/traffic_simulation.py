import numpy as np
from osdf.framework import ObjectiveDrivenDynamicalStochasticField


def run_traffic_simulation():
    # Define topology
    size = 3  # Grid size
    n_entities = size * size

    # Define configuration space
    config_space = {
        "intersection": ["NS_GREEN", "EW_GREEN", "ALL_RED"],
        "sensor": ["active", "inactive"]
    }

    # Define default transition rules
    transitions = {
        "NS_GREEN": {"ALL_RED": 1.0},
        "EW_GREEN": {"ALL_RED": 1.0},
        "ALL_RED": {"NS_GREEN": 0.5, "EW_GREEN": 0.3, "ALL_RED": 0.2}
    }

    # Build adjacency matrix (simple grid)
    adjacency_matrix = np.zeros((n_entities, n_entities))

    # Connect adjacent nodes (up/down/left/right)
    def connect(i, j):
        if i // size == j // size and abs(i - j) == 1:  # Same row
            return True
        if i % size == j % size and abs(i - j) == size:    # Same column
            return True
        return False
    
    for i in range(size * size):
        for j in range(size * size):
            if connect(i, j):
                adjacency_matrix[i][j] = 1.0
    
    # Generator matrix (symmetric transitions)
    valid_states = len(config_space["intersection"])
    generator = np.zeros((valid_states, valid_states))
    for i in range(valid_states):
        for j in range(valid_states):
            if i != j:
                generator[i][j] = 1.0
        generator[i][i] = -np.sum(generator[i])
    
    # Define entity types explicitly using id_to_type
    id_to_type = {
        f"{i}": "intersection" for i in range(size*size)
    }
    
    # Run simulation
    system = ObjectiveDrivenDynamicalStochasticField(
        config_space=config_space,
        transition_rules=transitions,
        adjacency_matrix=adjacency_matrix,
        generator=generator,
        env_entities=[],  # No environmental signals in this example
        acting_entities=[f"{i}" for i in range(9)],
        id_to_type=id_to_type,
        propagator_fn=None  # Uses default PQ formulation
    )

    # Set initial configurations safely
    initial_configs = {str(i): "ALL_RED" for i in range(n_entities)}
    system.config_manager.set_initial_states(initial_configs)
    
    # Simulate traffic flow
    system.run_simulation(steps=200, verbose=True)

    
    # Visualize results
    system.visualize_topology()
    system.visualize_trajectories([str(i) for i in range(5)])
    
if __name__ == "__main__":
    run_traffic_simulation()