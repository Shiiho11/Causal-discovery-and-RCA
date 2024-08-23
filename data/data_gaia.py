import json
import os

import networkx as nx
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
_path = script_dir + '/dataset/GAIA/'
_knowledge = 'Business flow: the front end opens the webpage, the webpage calls webservice, the webservice generates uuid, and calls redisservice to store the uuid in the redis cache, and set the time limit. Then, the webservice calls mobservice and loginservice at the same time, where mobservice simulates the mobile phone user scanning the QR code login action, and calls the redisservice service to store the user id information in the value in the corresponding uuid; during this period, the loginservice will be called every second Redisservice reads the complete information of the corresponding uuid until the uuid becomes invalid or the information is complete. If the information is complete, loginservice will call the login service dbservice, and dbservice will connect to mysql to obtain information such as the account corresponding to the user id, and generate a token to return. Of course, all services in the entire process are uniformly registered and discovered by zookeeper.'


def get_data():
    graph_data = pd.read_csv(_path + 'graph.csv').values
    true_dag = nx.DiGraph()
    for u, v in graph_data:
        true_dag.add_edge(u, v)
    target_node = 'webservice'
    normal_data = pd.read_csv(_path + 'normal_data.csv')
    anomalous_data = pd.read_csv(_path + 'anomalous_data.csv')
    labels = pd.read_csv(_path + 'labels_data.csv')
    return true_dag, target_node, normal_data, anomalous_data, labels


def get_info(knowledge=False):
    domain = 'Microservice Architecture Web Application'
    background = 'This is a web application with microservices architecture.' \
                 'The dataset records the latency of each server\'s response to requests. '\
                 'I want to find the root cause of the malfunction through a causal graph. '\
                 'The variable is the delay in the server\'s response to requests.'
    if knowledge:
        background = '\n'.join([background, _knowledge])
    var_info = dict()
    var_info_data = pd.read_csv(_path + 'var_info.csv').values
    for key, values in var_info_data:
        if str(values) == 'nan':
            var_info[key] = ''
        else:
            var_info[key] = values
    return domain, background, var_info


def get_llm_dag(knowledge=False):
    filename = '/bcgllm.json'
    llm_dag = nx.DiGraph()
    with open(script_dir + filename, 'r') as f:
        data = json.load(f)
        if knowledge:
            for dag_info in data:
                if dag_info['name'] == 'gaia_know':
                    llm_dag.add_nodes_from(dag_info['nodes'])
                    llm_dag.add_edges_from(dag_info['edges'])
                    break
        else:
            for dag_info in data:
                if dag_info['name'] == 'gaia':
                    llm_dag.add_nodes_from(dag_info['nodes'])
                    llm_dag.add_edges_from(dag_info['edges'])
                    break
    return llm_dag