import json
import os

import networkx as nx
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
_path = script_dir + '/dataset/sachs/'


def get_data():
    graph_data = pd.read_csv(_path + 'graph.csv').values
    true_dag = nx.DiGraph()
    for u, v in graph_data:
        true_dag.add_edge(u, v)
    target_node = 'akt'
    normal_data = pd.read_csv(_path + 'normal_data.csv')
    anomalous_data = pd.read_csv(_path + 'anomalous_data.csv')
    labels = pd.read_csv(_path + 'labels_data.csv')
    return true_dag, target_node, normal_data, anomalous_data, labels


def get_info():
    domain = 'biology'
    background = 'This dataset measures the expression level of different proteins and phospholipids in human cells. It includes the simultaneous measurements of 11 phosphorylated proteins and phospholipids derived from thousands of individual primary immune system cells, subjected to both general and specific molecular interventions.'
    var_info = dict()
    var_info_data = pd.read_csv(_path + 'var_info.csv').values
    for key, values in var_info_data:
        if str(values) == 'nan':
            var_info[key] = ''
        else:
            var_info[key] = values
    return domain, background, var_info


def get_llm_dag():
    filename = '/bcgllm.json'
    llm_dag = nx.DiGraph()
    with open(script_dir + filename, 'r') as f:
        data = json.load(f)
        for dag_info in data:
            if dag_info['name'] == 'sachs':
                llm_dag.add_nodes_from(dag_info['nodes'])
                llm_dag.add_edges_from(dag_info['edges'])
                break
    return llm_dag
