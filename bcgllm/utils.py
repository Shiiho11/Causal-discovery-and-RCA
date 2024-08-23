import json
import os


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"The folder '{directory}' has been successfully created!")
    else:
        print(f"The folder '{directory}' already exists.")

def save_graph(graph, filename):
    data = dict()
    data['nodes'] = list(graph.nodes)
    data['edges'] = list(graph.edges)
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(data, f)


def load_graph(graph_type, filename):
    graph = graph_type()
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        graph.add_nodes_from(data['nodes'])
        graph.add_edges_from(data['edges'])
    return graph
