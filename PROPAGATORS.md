# Choosing the Right Propagator

One of the most critical design decisions in your Objective-Driven Dynamical Stochastic Field (ODSF) implementation is the choice of **propagator function** (`P[Q]`) that governs how objective signals evolve and propagate across the field.

This decision directly impacts:
- The system's ability to perform **local gradient computation**
- The emergence of **coherent global behavior** from local interactions
- The framework's capacity for **adaptive learning and optimization**

---

## 🧠 Understanding the Propagator

The propagator defines how entities interact through:

```
Γx'x = Γx' * Px'x
```

Where:
- `Γ` is the objective signal tensor
- `Px'x` is the propagation operator between entities x' → x
- `G̃` is the adjoint generator
- `S` is the resolvent `(εI - G)^{-1}`
- `Π` is the projection onto the null space of `G`

As shown in Theorem 3.32, a proper propagator allows decentralized systems to compute gradients locally while still converging to globally optimal configurations.

---

## ✅ Available Propagator Types

| Propagator | Formula | Use Case |
|-----------|--------|----------|
| **Default P[Q]** | `P[Q] = 1 + G̃SΠ` | General-purpose local gradient computation |
| **Gradient Estimator** | Local finite-differences | When full dynamics are unknown or nonlinear |
| **Time-Forwarding** | `Φ = lim_{t→∞} e^{Gt}` | Long-term equilibrium forecasting |
| **Hebbian Learner** | Correlation-based updates | Biologically plausible signal adaptation |

---

## 🎯 How to Choose

### 1. If You Need **Local Gradient Computation**
Use default `P[Q] = 1 + G̃SΠ` where:
- `G̃` is the adjoint generator
- `S` is the resolvent
- `Π` projects onto null space

```python
from odsf.learning.signal_learner import build_pq_propagator

stationary = np.array([0.5, 0.5])
pq_propagator = build_pq_propagator(generator, stationary)

system = ObjectiveDrivenDynamicalStochasticField(
    # ... other params ...
    propagator_fn=pq_propagator
)
```

> According to Theorem 3.32, this ensures local gradient computations converge correctly

---

### 2. If You're Simulating **Long-Term Equilibrium**
Use time-forwarding formulation based on infinite-time limit:

```python
from odsf.learning.signal_learner import time_forward_propagator

system = ObjectiveDrivenDynamicalStochasticField(
    # ... other params ...
    propagator_fn=time_forward_propagator
)
```

This is useful when you want stability over responsiveness, such as modeling long-term infrastructure reliability.

---

### 3. If Using **Biological Plausibility**
For neuromorphic computing or brain-inspired architectures:

```python
from odsf.learning.signal_learner import hebbian_learning_propagator

system = ObjectiveDrivenDynamicalStochasticField(
    # ... other params ...
    propagator_fn=hebbian_learning_propagator
)
```

This propagator follows biological correlation-based update rules rather than analytical formulations.

---

### 4. If Learning Signal Transforms
When you want to learn an optimized propagator using deep learning:

```python
from odsf.learning.signal_learner import SignalLearningController

signal_learner = SignalLearningController(system.signal_propagator)
signal_learner.learn_signal_transformation(target_signals)

# Then use the learned transformation in new simulations
learned_propagator = signal_learner.transform_signal
```

This enables adaptive learning of propagation patterns based on real-world data.

---

## 🔁 Mathematical Background

The paper shows (Theorem 3.32) that when acting entities are strongly connected, **propagators of the form `P[Q]` enable local gradient computation** even though they encode global information via:

```
P[Q] = 1 + G̃ S Π
```

Where:
- `G̃` is the adjoint generator
- `S` is the resolvent of the infinitesimal generator
- `Π` is the projection onto null space of `G`

This is the recommended starting point for most applications, especially those requiring coordinated multi-agent optimization.

---

## 📌 Best Practices

- Start with `P[Q] = 1 + G̃SΠ` for general AI/optimization tasks
- For large-scale systems, consider propagator decomposition per entity type
- Always validate strong connectivity among acting entities
- Monitor sensitivity using `.compute_sensitivity()` method
- Regularly re-normalize weights with `.normalize_weights()`


