from typing import Dict
import numpy as np
from scipy.linalg import expm


class SignalLearningController:
    def __init__(self, signal_propagator: 'ObjectiveSignalPropagator', hidden_dim: int = 64):
        self.propagator = signal_propagator
        self.signal_network = self._build_network(hidden_dim)
        
    def _build_network(self, hidden_dim: int):
        return np.random.rand(
            self.propagator.stationary.shape[0], 
            self.propagator.stationary.shape[0]
        )
        
    def transform_signal(self, signal: np.ndarray) -> np.ndarray:
        return self.signal_network @ signal
        
    def learn_signal_transformation(self, target_signals: Dict[str, np.ndarray]):
        for entity_id, target in target_signals.items():
            current_signal = self.propagator.signals.get(entity_id, np.zeros_like(target))
            error = target - current_signal
            self.signal_network -= 0.01 * np.outer(error, current_signal)

def compute_resolvent(G: np.ndarray) -> np.ndarray:
    """Compute S = resolvent operator (εI - G)^{-1}"""
    epsilon = 1e-6  # Small regularization term
    try:
        return np.linalg.inv(epsilon * np.eye(*G.shape) - G)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if matrix is singular
        return np.linalg.pinv(epsilon * np.eye(*G.shape) - G)

def compute_projection(stationary: np.ndarray) -> np.ndarray:
    """Compute Π = projection onto null space of G"""
    outer = np.outer(stationary, np.ones_like(stationary))
    return outer / np.trace(outer)  # Normalize to ensure idempotence

def build_pq_propagator(G: np.ndarray, stationary: np.ndarray):
    """Build P[Q] operator given generator G and stationary distribution"""
    def pq_propagator(g: np.ndarray, signal: np.ndarray, pi: np.ndarray):
        """
        Actual propagator function implementing:
        P[Q] = 1 + G̃SΠ
        
        Parameters:
        g: Generator matrix for current entity
        signal: Input objective signal vector
        pi: Stationary distribution
        
        Returns:
        np.ndarray: Transformed objective signal
        """
        # Compute resolvent
        S = compute_resolvent(g)
        
        # Ensure numerical stability by projecting onto null space
        Pi = compute_projection(pi)
        
        # Build propagator using theoretical formula
        I = np.eye(*g.shape)
        P = I + g.T @ S @ Pi
        
        # Apply transformation to input signal
        return P @ signal
    
    return pq_propagator

def hebbian_learning_propagator(g: np.ndarray, signal: np.ndarray, pi: np.ndarray):
    """Uses Hebbian-like update rule for signal propagation"""
    learning_rate = 0.01
    new_weights = g.copy()
    
    # Update weights based on correlation between inputs and outputs
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            new_weights[i,j] += learning_rate * signal[i] * pi[j]
    
    # Normalize row sums to maintain Markov property
    new_weights -= np.diag(np.sum(new_weights, axis=1))
    return new_weights

def time_forward_propagator(g: np.ndarray, signal: np.ndarray, pi: np.ndarray):
    """Implements Φ = lim_{t→∞} e^{Gt}"""
    dt = 0.01
    steps = 1000
    P = expm(g * dt)
    
    # Forward simulation
    dist = signal.copy() / signal.sum()
    for _ in range(steps):
        dist = dist @ P
    
    return dist

def gradient_estimating_propagator(g: np.ndarray, signal: np.ndarray, pi: np.ndarray):
    """Estimate gradient information during propagation"""
    sensitivity = np.zeros_like(signal)
    
    # Perturb each state and measure response
    for i in range(len(signal)):
        perturbed = signal.copy()
        perturbed[i] += 1e-5
        
        # Measure change in final objective
        delta_pi = np.abs(perturbed - pi).mean()
        grad_i = (perturbed - signal).dot(pi) / 1e-5
        sensitivity[i] = grad_i
    
    return sensitivity