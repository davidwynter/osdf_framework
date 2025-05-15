import numpy as np
from osdf import ConfigurationSpaceManager
from osdf import DynamicalEvolutionEngine
from osdf import EntityInteractionEngine
from osdf import EnvironmentGraphModeler
from osdf import ObjectiveSignalPropagator
from osdf import ObservationLogger
from osdf.learning.signal_learner import build_pq_propagator
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Callable


class ObjectiveDrivenDynamicalStochasticField:
    """
    A unified framework implementing the principles of complete configuration,
    locality, and purposefulness from the paper on objective-driven dynamical
    stochastic fields.
    
    This class integrates all components into a cohesive system that can be evolved
    over time while maintaining theoretical consistency with the mathematical
    formulation in the paper.
    """
    
    def __init__(self, 
                 config_space: Dict[str, List],
                 transition_rules: Dict[str, Dict],
                 adjacency_matrix: np.ndarray,
                 generator: np.ndarray,
                 env_entities: List[str],
                 acting_entities: List[str],
                 id_to_type: Optional[Dict[str, str]] = None,
                 propagator_fn: Optional[Callable] = None
                 ):
        """
        Initialize the unified framework with all necessary components.
        
        Parameters:
        -----------
        config_space : Dict[str, List]
            Dictionary mapping entity types to their possible configurations
        transition_rules : Dict[str, Dict]
            Markov transition rules for each configuration state
        adjacency_matrix : np.ndarray
            NxN matrix representing spatial interactions between entities
        generator : np.ndarray
            Infinitesimal generator matrix G for continuous-time dynamics
        env_entities : List[str]
            List of environmental entity IDs
        acting_entities : List[str]
            List of acting entity IDs
        id_to_type: Map from entity ID to type (default: inferred via alpha-characters)
        propagator_fn: Custom function P(x'|x) for signal transformation, Custom function for signal propagation (default: uses built-in formulation)
        """
        # Initialize core components
        self.config_manager = ConfigurationSpaceManager(config_space, transition_rules)
        self.graph_modeler = self._initialize_graph(adjacency_matrix)
        # Use explicit mapping or fallback to default schema
        self.id_to_type = id_to_type or {
            str(i): ''.join(filter(str.isalpha, str(i))) or "default"
            for i in range(adjacency_matrix.shape[0])
        }
        self.dyn_engine = DynamicalEvolutionEngine(generator)
        self.config_manager.validate_initialization(list(self.graph_modeler.graph.nodes))        
        
        # Create PQ-based propagator
        stationary_distribution = np.array([0.5, 0.5])
        # Initialize objective signal propagator
        if propagator_fn is None:
            from .learning.signal_learner import build_pq_propagator
            self.propagator_fn = build_pq_propagator(self.dyn_engine.G, self.dyn_engine.current_distribution)
        else:
            self.propagator_fn = propagator_fn

        # Initialize interaction and objective components
        self.adj_weights = self._create_adjacency_weights(adjacency_matrix)
        
        self.signal_propagator = ObjectiveSignalPropagator(
            environment_entities=env_entities,
            acting_entities=acting_entities,
            adjacency_weights=self.adj_weights,
            dynamics_engine=self.dyn_engine,
            propagator_fn=self.propagator_fn
        )
        
        self.interaction_engine = EntityInteractionEngine(
            self.config_manager, 
            self.graph_modeler, 
            self.signal_propagator
        )
        
        # Initialize logging
        self.logger = ObservationLogger()
        
        # System state tracking
        self.timestep = 0
        self.converged = False
        self.stability_threshold = 1e-5
        self.stationary_history = []
    
    def _initialize_graph(self, adjacency: np.ndarray) -> EnvironmentGraphModeler:
        """Initialize the graph structure based on the adjacency matrix"""
        n = adjacency.shape[0]
        graph = nx.Graph()
        
        # Add nodes for each entity
        for i in range(n):
            graph.add_node(str(i))
        
        # Add edges based on adjacency matrix
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] > 0:
                    graph.add_edge(str(i), str(j), weight=adjacency[i, j])
        
        return EnvironmentGraphModeler(graph)
    
    def _create_adjacency_weights(self, adjacency: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert adjacency matrix to weighted neighbor dictionary"""
        weights = {}
        for i in range(adjacency.shape[0]):
            node_id = str(i)
            neighbors = {
                str(j): float(adjacency[i,j]) 
                for j in range(adjacency.shape[0]) 
                if adjacency[i,j] > 0
            }
            if neighbors:
                total = sum(neighbors.values())
                weights[node_id] = {k: v/total for k,v in neighbors.items()}
            else:
                weights[node_id] = {}
        return weights
    
    def set_initial_states(self, initial_configs: Dict[str, str]):
        """Set initial configurations for entities"""
        for entity_id, config in initial_configs.items():
            self.config_manager.set_initial_state(entity_id, config)
    
    def normalize_weights(self):
        """Normalize adjacency weights so they sum to 1 for each entity"""
        for entity_id in self.adj_weights:
            neighbor_dict = self.adj_weights[entity_id]
            total = sum(neighbor_dict.values())
            
            if total > 0:
                self.adj_weights[entity_id] = {
                    nid: weight/total for nid, weight in neighbor_dict.items()
                }
    
    def step(self):
        """Perform one full simulation step"""
        # 1. Propagate and update entity states
        self.interaction_engine.propagate_and_update()
        
        # 2. Log current state
        actions = self.interaction_engine.build_entity_action_map()
        objectives = {eid: self.signal_propagator.get_effective_objective(eid) 
                     for eid in self.graph_modeler.graph.nodes}
        
        self.logger.log_timestep(self.timestep, actions, objectives)
        
        # 3. Track stationary distribution for convergence detection
        current_dist = self.config_manager.build_distribution()
        self.stationary_history.append(current_dist.copy())
        
        # 4. Check for convergence
        if len(self.stationary_history) > 1:
            dist_change = np.abs(current_dist - self.stationary_history[-2]).mean()
            self.converged = dist_change < self.stability_threshold
        
        # 5. Increment timestep
        self.timestep += 1
    
    def run_simulation(self, steps: int = 100, verbose: bool = False):
        """Run the simulation loop for specified number of steps"""
        print(f"Starting simulation with {steps} steps...")
        
        for step_idx in range(steps):
            # Perform interaction + signal propagation
            self.interaction_engine.propagate_and_update()
            
            # Log observations
            actions = self.get_current_configurations()
            objectives = self.get_current_objectives()
            self.logger.log_timestep(step_idx, actions, objectives)
            
            if verbose and step_idx % 10 == 0:
                print(f"Step {step_idx}: Current configurations:")
                print(actions)
                    
    def get_current_configurations(self) -> Dict[str, str]:
        """Get current configuration of all entities"""
        return {
            eid: self.config_manager.get_current_config(eid)
            for eid in self.graph_modeler.graph.nodes
        }
    
    def get_current_objectives(self) -> Dict[str, float]:
        """Get current objective values for all acting entities"""
        return {
            eid: self.signal_propagator.get_effective_objective(eid)
            for eid in self.signal_propagator.act_entities
        }
    
    def export_observations(self) -> pd.DataFrame:
        """Export logged observations as a DataFrame"""
        return self.logger.export_log()
    
    def visualize_topology(self):
        """Visualize the interaction topology"""
        self.graph_modeler.visualize()
    
    def visualize_trajectories(self, entity_ids: List[str], n_steps: int = None):
        """Plot configuration trajectories over time"""
        self.logger.visualize_actions(entity_ids, n_steps or self.timestep)
    
    def compute_sensitivity(self, param: str, delta: float = 1e-5) -> Dict[str, float]:
        """
        Compute sensitivity of the system to parameter changes using Definition 3.26
        
        Parameters:
        -----------
        param : str
            Parameter to perturb ('transition' or 'generator')
        delta : float
            Small perturbation value
            
        Returns:
        --------
        Dict[str, float]
            Sensitivity of each entity's objective to the parameter change
        """
        original_obj = self.get_current_objectives()
        
        # Perturb the parameter
        if param == 'transition':
            orig_transitions = {k: v.copy() for k, v in self.config_manager.transitions.items()}
            # Apply small random perturbation to transitions
            for state in self.config_manager.transitions:
                noise = np.random.normal(0, delta, len(self.config_manager.transitions[state]))
                new_probs = np.array(list(self.config_manager.transitions[state].values())) + noise
                new_probs = np.clip(new_probs, 0, 1)
                new_probs /= new_probs.sum()
                
                self.config_manager.transitions[state] = {
                    k: p for k, p in zip(self.config_manager.transitions[state].keys(), new_probs)
                }
                
        elif param == 'generator':
            orig_generator = self.dyn_engine.G.copy()
            # Apply perturbation to generator matrix
            perturbation = np.random.normal(0, delta, orig_generator.shape)
            self.dyn_engine.G += perturbation
            
        else:
            raise ValueError(f"Unknown parameter type: {param}")
        
        # Run simulation to observe effect
        self.run_simulation(10, verbose=False)
        perturbed_obj = self.get_current_objectives()
        
        # Restore original parameters
        if param == 'transition':
            self.config_manager.transitions = orig_transitions
        elif param == 'generator':
            self.dyn_engine.G = orig_generator
        
        # Calculate sensitivities
        sensitivities = {}
        for eid in original_obj:
            sens = (perturbed_obj[eid] - original_obj[eid]) / delta
            sensitivities[eid] = float(sens)
        
        return sensitivities
    
    def verify_local_objective_property(self) -> bool:
        """
        Verify the local objective property from Proposition 3.26
        
        Returns:
        --------
        bool
            Whether all acting entities have the same objective value
        """
        objectives = list(self.get_current_objectives().values())
        return np.std(objectives) < self.stability_threshold
