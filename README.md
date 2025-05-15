# Objective-Driven Dynamical Stochastic Fields (ODSF)

A framework implementing the mathematical theory of objective-driven dynamical stochastic fields described in [this paper](https://arxiv.org/abs/2504.16115v1).

## 🔍 Overview

This framework implements a theoretical system based on three fundamental principles:
1. **Complete Configuration** - Continuous-time Markovian dynamics
2. **Locality** - Local interaction constraints
3. **Purposefulness** - Objective-driven behavior optimization

![Space-Time Diagram](docs/images/spacetime-diagram.png)

## 🧠 Key Features

- Modular implementation of the mathematical framework
- Support for both discrete and continuous-time simulations
- Neural network integration for adaptive learning
- Graph-based spatial modeling
- Comprehensive observability and logging
- Gradient-based optimization of objective signals

## 📦 Installation

```bash
# Install via pip (recommended)
pip install odsf

# Or install from source
git clone https://github.com/yourname/odsf.git
cd odsf
pip install -e .
```

## 🚀 Quick Start
```
from odsf.framework import ObjectiveDrivenDynamicalStochasticField
from odsf.core.configuration_manager import ConfigurationSpaceManager
from odsf.visualization.logger import ObservationLogger

# Define system parameters
config_space = {"agent": ["state0", "state1"]}
transitions = {
    "state0": {"state1": 0.5, "state0": 0.5},
    "state1": {"state0": 0.7, "state1": 0.3}
}
adjacency_matrix = np.array([[0, 1], [1, 0]])
generator = np.array([[-1, 1], [1, -1]])

# Initialize the system
system = ObjectiveDrivenDynamicalStochasticField(
    config_space=config_space,
    transition_rules=transitions,
    adjacency_matrix=adjacency_matrix,
    generator=generator,
    env_entities=["0"],
    acting_entities=["1"]
)

# Run simulation
system.run_simulation(steps=100)

# Export and visualize results
df = system.export_observations()
print(df.head())
system.visualize_topology()
system.visualize_trajectories(["0", "1"])
```