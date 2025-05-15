from typing import Callable, List, Dict, Optional
import numpy as np

from osdf import DynamicalEvolutionEngine

class ObjectiveSignalPropagator:
    def __init__(
        self,
        environment_entities: List[str],
        acting_entities: List[str],
        adjacency_weights: Dict[str, Dict[str, float]],
        dynamics_engine: 'DynamicalEvolutionEngine',
        propagator_fn: Optional[Callable] = None
    ):
        self.env_entities = environment_entities
        self.act_entities = acting_entities
        self.adjacency = adjacency_weights
        self.dynamics = dynamics_engine
        self.signals = {}
        self.stationary = dynamics_engine.current_distribution

    def compute_propagation_matrix(self) -> np.ndarray:
        epsilon = 1e-6
        try:
            reg_matrix = epsilon * np.eye(*self.dynamics.G.shape) + 1e-8 * np.random.randn(*self.dynamics.G.shape)
            return np.linalg.inv(reg_matrix - self.dynamics.G)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(reg_matrix - self.dynamics.G)

    def propagate_signals(self):
        new_signals = {}

        for entity in self.act_entities:
            weighted_sum = np.zeros_like(self.stationary)
            neighbors = self.adjacency.get(entity, {})

            for neighbor, weight in neighbors.items():
                signal = self.signals.get(neighbor, np.zeros_like(self.stationary))
                propagated = self._propagate(signal)
                weighted_sum += weight * propagated

            new_signals[entity] = weighted_sum

        self.signals.update(new_signals)

    def _propagate(self, signal: np.ndarray) -> np.ndarray:
        return signal * 0.9  # Simple decay model for demonstration