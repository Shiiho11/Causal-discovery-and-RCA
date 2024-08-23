import json

import pandas as pd

from bcgllm.ccg import CCG
from bcgllm.utils import create_directory_if_not_exists
from data import data_auto_mpg, data_gaia, data_sachs
from mcts.calculate_utils import CausalMechanismReusableSetter, IndependenceTester, calculate_dag_score
from mcts.graph_utils import get_ordered_node_list
from bcgllm.llm import DeepSeekClient, GPTClient


def exp_auto_mpg(client):
    true_dag, target_node, normal_data, anomalous_data, labels = data_auto_mpg.get_data()
    domain, background, var_info = data_auto_mpg.get_info()
    normal_data.drop(columns=['cylinders', 'modelyear', 'origin'], inplace=True)
    anomalous_data.drop(columns=['cylinders', 'modelyear', 'origin'], inplace=True)

    ccg = CCG(llm_client=client, domain=domain, variables_description=var_info, domain_expert=None,
              background=background, start_node=target_node, full_visit=True)
    dag = ccg.construct()
    print(dag)
    print(dag.edges)

    dag_score_dict = dict()
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(true_dag))

    dag_record_list = []
    calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
                        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                        dag_record_list=dag_record_list,
                        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                        true_dag=true_dag, given_dag=true_dag,
                        num_distribution_samples=1500)
    dag_record_list[-1]['name'] = 'auto_mpg'
    return dag_record_list


def exp_gaia(client):
    true_dag, target_node, normal_data, anomalous_data, labels = data_gaia.get_data()
    domain, background, var_info = data_gaia.get_info(knowledge=False)
    normal_data = normal_data.head(1000)
    anomalous_data = anomalous_data.head(100)
    labels = labels.head(100)
    normal_data.drop(columns=['time'], inplace=True)
    anomalous_data.drop(columns=['time'], inplace=True)

    ccg = CCG(llm_client=client, domain=domain, variables_description=var_info, domain_expert=None,
              background=background, start_node=target_node, full_visit=True)
    dag = ccg.construct()
    print(dag)
    print(dag.edges)

    dag_score_dict = dict()
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(true_dag))

    dag_record_list = []
    calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
                        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                        dag_record_list=dag_record_list,
                        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                        true_dag=true_dag, given_dag=true_dag,
                        num_distribution_samples=1500)
    dag_record_list[-1]['name'] = 'gaia'

    #
    domain, background, var_info = data_gaia.get_info(knowledge=False)
    ccg = CCG(llm_client=client, domain=domain, variables_description=var_info, domain_expert=None,
              background=background, start_node=target_node, full_visit=True)
    dag = ccg.construct()
    print(dag)
    print(dag.edges)
    calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
                        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                        dag_record_list=dag_record_list,
                        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                        true_dag=true_dag, given_dag=true_dag,
                        num_distribution_samples=1500)
    dag_record_list[-1]['name'] = 'gaia_know'

    return dag_record_list


def exp_sachs(client):
    true_dag, target_node, normal_data, anomalous_data, labels = data_sachs.get_data()
    domain, background, var_info = data_sachs.get_info()

    ccg = CCG(llm_client=client, domain=domain, variables_description=var_info, domain_expert=None,
              background=background, start_node=target_node, full_visit=True)
    dag = ccg.construct()
    print(dag)
    print(dag.edges)

    dag_score_dict = dict()
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(true_dag))

    dag_record_list = []
    calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
                        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                        dag_record_list=dag_record_list,
                        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                        true_dag=true_dag, given_dag=true_dag,
                        num_distribution_samples=1500)
    dag_record_list[-1]['name'] = 'sachs'
    return dag_record_list


def exp_bcgllm():
    output_path = '../result/'
    create_directory_if_not_exists(output_path)

    api_key = ''
    base_url = ''
    client = GPTClient(api_key, base_url)
    client.temperature = 0

    record_auto_mpg = exp_auto_mpg(client)
    record_gaia = exp_gaia(client)
    record_sachs = exp_sachs(client)
    dag_record_list = record_auto_mpg + record_gaia + record_sachs

    with open(output_path + 'bcgllm.json', 'w', encoding='utf-8') as f:
        json.dump(dag_record_list, f)
    dag_record_csv_data = []
    for dag_info in dag_record_list:
        dag_record_csv_data.append(
            [dag_info['index'], dag_info['name'], dag_info['score'], dag_info['RCA_loss'], dag_info['RCA_accuracy'],
             dag_info['RCA_precision'], dag_info['RCA_recall'], dag_info['RCA_F1'], dag_info['graph_accuracy'],
             dag_info['graph_precision'], dag_info['graph_recall'], dag_info['graph_F1'],
             dag_info['NHD_true'], dag_info['NHD_given'], dag_info['NHD_empty'],
             dag_info['RCA_precision_micro'], dag_info['RCA_recall_micro'], dag_info['RCA_F1_micro'],
             dag_info['nodes'], dag_info['edges']])
    pd.DataFrame(dag_record_csv_data,
                 columns=['index', 'name', 'score', 'RCA_loss', 'RCA_accuracy', 'RCA_precision', 'RCA_recall', 'RCA_F1',
                          'graph_accuracy', 'graph_precision', 'graph_recall', 'graph_F1', 'NHD_true', 'NHD_given', 'NHD_empty',
                          'RCA_precision_micro', 'RCA_recall_micro', 'RCA_F1_micro',
                          'nodes', 'edges'
                          ]).to_csv(output_path + 'bcgllm.csv', index=False)


if __name__ == '__main__':
    exp_bcgllm()
