import networkx as nx
from typing import Dict, List
import matplotlib.pyplot as plt

class EnvironmentGraphModeler:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def add_entities(self, entities: Dict[str, str]):
        for node_id, entity_type in entities.items():
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]['type'] = entity_type
            else:
                self.graph.add_node(node_id, type=entity_type)

    def get_neighbors(self, entity_id: str) -> List[str]:
        return list(self.graph.neighbors(entity_id))

    def is_connected(self, entity_a: str, entity_b: str) -> bool:
        return self.graph.has_edge(entity_a, entity_b)

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        labels = nx.get_node_attributes(self.graph, 'type')
        nx.draw(self.graph, pos, with_labels=True, labels=labels, node_size=800)
        plt.show()