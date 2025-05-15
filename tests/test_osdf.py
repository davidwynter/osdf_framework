import pytest
import numpy as np
import networkx as nx
from osdf.core.configuration_manager import ConfigurationSpaceManager
from osdf.core.dynamical_engine import DynamicalEvolutionEngine
from osdf.core.graph_modeler import EnvironmentGraphModeler
from osdf.learning.signal_learner import SignalLearningController
from osdf.agents.adaptive_entity import AdaptiveEntity
from osdf.visualisation.logger import ObservationLogger

# Fixtures
@pytest.fixture
def config_space():
    return {"A": ["a0", "a1"], "B": ["b0", "b1"]}

@pytest.fixture
def transitions():
    return {
        "a0": {"a1": 0.5, "a0": 0.5},
        "a1": {"a0": 0.7, "a1": 0.3}
    }

@pytest.fixture
def generator_matrix():
    return np.array([[-1, 1], [1, -1]])

@pytest.fixture
def adjacency_matrix():
    return np.array([[0, 1], [1, 0]])

@pytest.fixture
def stationary_distribution():
    return np.array([0.5, 0.5])

# ConfigurationSpaceManager Tests
def test_config_manager_init(config_space, transitions):
    csm = ConfigurationSpaceManager(config_space, transitions)
    assert csm.config_space == config_space
    assert csm.transitions == transitions

def test_set_initial_state(config_space, transitions):
    csm = ConfigurationSpaceManager(config_space, transitions)
    csm.set_initial_state("A1", "a0")
    assert csm.get_current_config("A1") == "a0"

def test_invalid_configuration(config_space, transitions):
    csm = ConfigurationSpaceManager(config_space, transitions)
    with pytest.raises(ValueError):
        csm.set_initial_state("A1", "invalid")

# DynamicalEvolutionEngine Tests
def test_generator_validation(generator_matrix):
    dee = DynamicalEvolutionEngine(generator_matrix)
    assert np.array_equal(dee.G, generator_matrix)

def test_evolution_step(generator_matrix):
    dee = DynamicalEvolutionEngine(generator_matrix)
    initial_dist = np.array([1.0, 0.0])
    dee.current_distribution = initial_dist
    new_dist = dee.step(0.1)
    assert np.isclose(new_dist.sum(), 1.0)
    assert not np.array_equal(new_dist, initial_dist)

def test_invalid_generator():
    with pytest.raises(ValueError):
        # Invalid generator (rows don't sum to zero)
        bad_generator = np.array([[1, 1], [1, -1]])
        DynamicalEvolutionEngine(bad_generator)

# EnvironmentGraphModeler Tests
def test_graph_creation(adjacency_matrix):
    graph_modeler = EnvironmentGraphModeler(nx.Graph())
    assert isinstance(graph_modeler.graph, nx.Graph)

def test_get_neighbors(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(str(i))
    
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0:
                graph.add_edge(str(i), str(j))
    
    graph_modeler = EnvironmentGraphModeler(graph)
    neighbors = graph_modeler.get_neighbors("0")
    assert "1" in neighbors

# Signal Learning Controller Tests
def test_signal_transformation(stationary_distribution):
    generator = np.array([[-1, 1], [1, -1]])
    dyn_engine = DynamicalEvolutionEngine(generator)
    propagator = SignalLearningController(dyn_engine)
    
    signal_input = np.array([1.0, 0.5])
    transformed = propagator.transform_signal(signal_input)
    assert transformed.shape == (2,)
    assert not np.array_equal(transformed, signal_input)

# Adaptive Entity Tests
def test_adaptive_entity():
    entity = AdaptiveEntity("E1", ["config1", "config2"])
    context = {
        "neighbors": [],
        "signals": 0.5,
        "current_config": "config1"
    }
    
    config = entity.predict_configuration(context)
    assert config in ["config1", "config2"]
    
    # Test learning step doesn't crash
    try:
        entity.learn(context, reward=1.0)
    except Exception as e:
        pytest.fail(f"Learning step failed with error: {str(e)}")

# Observation Logger Tests
def test_logging():
    logger = ObservationLogger()
    actions = {"E1": "config1", "E2": "config2"}
    objectives = {"E1": 0.8, "E2": 0.6}
    
    logger.log_timestep(0, actions, objectives)
    df = logger.export_log()
    
    assert len(df) == 1
    assert "action_E1" in df.columns
    assert "objective_E2" in df.columns