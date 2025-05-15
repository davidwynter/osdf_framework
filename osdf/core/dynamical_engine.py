import numpy as np
from scipy.linalg import expm
from ..utils.validation import validate_generator

class DynamicalEvolutionEngine:
    def __init__(self, generator: np.ndarray):
        if not validate_generator(generator):
            raise ValueError("Invalid generator matrix")
        
        self.G = generator
        self.num_states = generator.shape[0]
        self.current_distribution = np.ones(self.num_states) / self.num_states

    def step(self, dt: float):
        P = expm(self.G * dt)
        self.current_distribution = self.current_distribution @ P
        return self.current_distribution

    def reset(self, distribution: np.ndarray = None):
        if distribution is None:
            self.current_distribution = np.ones(self.num_states) / self.num_states
        else:
            self.current_distribution = distribution / distribution.sum()