import os

import networkx as nx
import pandas as pd

from data.utils.generate_data import generate_data_threshold, generate_data_rate

script_dir = os.path.dirname(os.path.realpath(__file__))

graph_file_name = script_dir + '/../../origin/GAIA/graph.csv'
file_name = script_dir + '/../../origin/GAIA/data.csv'


def run():
    graph_data = pd.read_csv(graph_file_name).values
    dag = nx.DiGraph()
    for u, v in graph_data:
        dag.add_edge(u, v)
    data = pd.read_csv(file_name)
    data.dropna(inplace=True)
    normal_data, anomalous_data, labels_data = generate_data_threshold(dag, data, 'webservice',
                                                                       abnormal_threshold_high=2,
                                                                       normal_data_size=5000, anomalous_data_size=500)
    print(normal_data)
    print(anomalous_data)
    print(labels_data)
    normal_data.to_csv(script_dir + '/../../dataset/GAIA/normal_data.csv', index=False)
    anomalous_data.to_csv(script_dir + '/../../dataset/GAIA/anomalous_data.csv', index=False)
    labels_data.to_csv(script_dir + '/../../dataset/GAIA/labels_data.csv', index=False)
