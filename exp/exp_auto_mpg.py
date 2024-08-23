import random

import networkx as nx
import pandas as pd

from data import data_auto_mpg
from exp import exp, get_other_methods_dag
from mcts.calculate_utils import CausalMechanismReusableSetter, IndependenceTester, calculate_dag_score, \
    save_other_methods_result
from mcts.graph_utils import get_ordered_node_list
from exp import create_directory_if_not_exists


def exp_auto_mpg():
    path = '../result/auto_mpg/'
    create_directory_if_not_exists(path)

    true_dag, target_node, normal_data, anomalous_data, labels = data_auto_mpg.get_data()
    domain, background, var_info = data_auto_mpg.get_info()
    llm_dag = data_auto_mpg.get_llm_dag()
    normal_data.drop(columns=['cylinders', 'modelyear', 'origin'], inplace=True)
    anomalous_data.drop(columns=['cylinders', 'modelyear', 'origin'], inplace=True)

    other_methods_dag = get_other_methods_dag(pd.concat([normal_data, anomalous_data]))
    pc_dag = other_methods_dag['pc']

    empty_graph = nx.DiGraph()
    nodes = list(true_dag.nodes)
    random.shuffle(nodes)
    empty_graph.add_nodes_from(nodes)

    dag_score_dict = dict()
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(true_dag))
    independence_tester = IndependenceTester(pd.concat([normal_data, anomalous_data]), get_ordered_node_list(true_dag))

    # temp_list = []
    # for i in range(50):
    #     print(i)
    #     calculate_dag_score(dag=true_dag, target_node=target_node, normal_data=normal_data,
    #                         anomalous_data=anomalous_data, labels=labels, dag_score_dict=dict(),
    #                         dag_record_list=temp_list,
    #                         causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
    #                         true_dag=true_dag, given_dag=true_dag,
    #                         num_distribution_samples=1000)
    # save_other_methods_result(temp_list)
    # exit()

    other_methods_record_list = []
    for name, dag in other_methods_dag.items():
        calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
                            anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                            dag_record_list=other_methods_record_list,
                            causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                            true_dag=true_dag, given_dag=dag,
                            num_distribution_samples=1000)
        other_methods_record_list[-1]['name'] = name
    save_other_methods_result(other_methods_record_list, path)

    iterationLimit = 100

    exp(output_path=path + 'mcts_pc_dag/', true_dag=true_dag, given_dag=pc_dag, target_node=target_node,
        normal_data=normal_data,
        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter, ops_add=True, ops_remove=True,
        ops_reverse=True, cause_node_only=False, num_distribution_samples=1000,
        pruning=False, llm_dag=llm_dag,
        independence_tester=independence_tester,
        iterationLimit=iterationLimit, timeLimit=None)

    exp(output_path=path + 'mcts_llm_pc_dag/', true_dag=true_dag, given_dag=pc_dag, target_node=target_node,
        normal_data=normal_data,
        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter, ops_add=True, ops_remove=True,
        ops_reverse=True, cause_node_only=False, num_distribution_samples=1000,
        pruning=True, llm_dag=llm_dag,
        independence_tester=independence_tester,
        iterationLimit=iterationLimit, timeLimit=None)

    exp(output_path=path + 'mcts_empty_graph/', true_dag=true_dag, given_dag=empty_graph, target_node=target_node,
        normal_data=normal_data,
        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter, ops_add=True, ops_remove=False,
        ops_reverse=False, cause_node_only=False, num_distribution_samples=1000,
        pruning=False, llm_dag=llm_dag,
        independence_tester=independence_tester,
        iterationLimit=iterationLimit, timeLimit=None)

    exp(output_path=path + 'mcts_llm_empty_graph/', true_dag=true_dag, given_dag=empty_graph, target_node=target_node,
        normal_data=normal_data,
        anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
        causal_mechanism_reusable_setter=causal_mechanism_reusable_setter, ops_add=True, ops_remove=False,
        ops_reverse=False, cause_node_only=False, num_distribution_samples=1000,
        pruning=True, llm_dag=llm_dag,
        independence_tester=independence_tester,
        iterationLimit=iterationLimit, timeLimit=None)


if __name__ == '__main__':
    exp_auto_mpg()
