# Main package initialization
from .core.configuration_manager import ConfigurationSpaceManager
from .core.graph_modeler import EnvironmentGraphModeler
from .core.dynamical_engine import DynamicalEvolutionEngine
from .core.entity_interaction import EntityInteractionEngine # type: ignore
from .agents.entity import Entity
from .agents.adaptive_entity import AdaptiveEntity
from .agents.adaptive_entity import NeuralNetworkAdapter
from .learning.signal_learner import SignalLearningController
from .visualisation.logger import ObservationLogger
from osdf.learning.neural_network import ObjectiveSignalPropagator

__version__ = "0.1.0"