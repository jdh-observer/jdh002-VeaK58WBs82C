from typing import Dict, List

from itertools import combinations
import networkx as nx
from networkx import Graph


# COLOR_MAP = {
#     'organisation': 'black',
#     'academic': 'blue',
#     'technocrat': 'red',
#     'diplomat': 'maroon',
#     'unknown': 'yellow',
#     '':'yellow'
# }


def generate_graph():
    return nx.Graph()


def add_entities(graph: Graph, entities: List[Dict[str, any]]):
    nodes = [(entity['entity_name'], {'name':entity['entity_name'],
                                      'entity_type':entity['entity_type']}) for entity in entities]
    graph.add_nodes_from(nodes)


def add_record_links(graph: Graph, record_entities: List[Dict[str, any]]):
    node_labels = [entity['entity_name'] for entity in record_entities]
    for node1, node2 in combinations(node_labels, 2):
        # print('adding link', node1, node2)
        graph.add_edge(node1, node2)


