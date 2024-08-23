import os

import networkx as nx
import pandas as pd

from data.utils.generate_data import generate_data_threshold, generate_data_rate

script_dir = os.path.dirname(os.path.realpath(__file__))

graph_file_name = script_dir + '/../../origin/sachs/graph.csv'
file_name = script_dir + '/../../origin/sachs/sachs.continuous.csv'


def run():
    graph_data = pd.read_csv(graph_file_name).values
    dag = nx.DiGraph()
    for u, v in graph_data:
        dag.add_edge(u, v)
    # plot(dag)
    data = pd.read_csv(file_name)
    normal_data, anomalous_data, labels_data = generate_data_rate(dag, data, 'akt',
                                                                  abnormal_rate_low=0.05, abnormal_rate_high=0.05,
                                                                  normal_data_size=1000, anomalous_data_size=100)
    print(normal_data)
    print(anomalous_data)
    print(labels_data)
    normal_data.to_csv(script_dir + '/../../dataset/sachs/normal_data.csv', index=False)
    anomalous_data.to_csv(script_dir + '/../../dataset/sachs/anomalous_data.csv', index=False)
    labels_data.to_csv(script_dir + '/../../dataset/sachs/labels_data.csv', index=False)
