import os
import random
import time
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.PermutationBased.GRaSP import grasp
from mcts.calculate_utils import save_result, calculate_dag_score
from mcts.graph_utils import dag_vars_num_to_name, GeneralGraph_to_DiGraph, adjacency_matrix_to_dag, has_cycle
from mcts.mcts import mcts, rewardPolicy
from mcts.state import State


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"The folder '{directory}' has been successfully created!")
    else:
        print(f"The folder '{directory}' already exists.")


def my_pc(normal_data):
    cg = pc(normal_data.values)
    cg.to_nx_graph()
    pdag = cg.nx_graph
    pdag = dag_vars_num_to_name(pdag, [str(name) for name in normal_data.columns])
    nd_edges_num = 0
    for (u, v) in pdag.edges:
        if (v, u) in pdag.edges:
            nd_edges_num += 1
    nd_edges_num = int(nd_edges_num / 2)
    print(pdag)
    print(pdag.edges)
    print(nd_edges_num)
    dag = pdag.copy()
    while has_cycle(dag):
        dag = pdag.copy()
        for (u, v) in pdag.edges:
            if (u, v) in dag.edges and (v, u) in dag.edges:
                (node1, node2) = random.choice([(u, v), (v, u)])
                dag.remove_edge(node1, node2)
    print('PC')
    print(dag)
    print(dag.edges)
    return dag


def my_ges(normal_data):
    Record = ges(normal_data.values)
    g = Record['G']
    dag = GeneralGraph_to_DiGraph(g)
    dag = dag_vars_num_to_name(dag, [str(name) for name in normal_data.columns])
    print('GES')
    print(dag)
    print(dag.edges)
    return dag


def my_ICALiNGAM(normal_data, lower_limit=0.01):
    model = lingam.ICALiNGAM()
    model.fit(normal_data)
    # print(model.causal_order_)
    # print(model.adjacency_matrix_)
    # from causallearn.search.FCMBased.lingam.utils import make_dot
    # labels = [f'{col}' for i, col in enumerate(normal_data.columns)]
    # g = make_dot(model.adjacency_matrix_, labels=labels)
    dag = adjacency_matrix_to_dag(model.adjacency_matrix_, list(normal_data.columns), lower_limit)
    print('ICALiNGAM')
    print(dag)
    print(dag.edges)
    return dag


# def my_GIN(normal_data):
#     raise Exception('ERROR')
#     G, K = GIN(normal_data)
#     dag = GeneralGraph_to_DiGraph(G)
#     dag = dag_vars_num_to_name(dag, [str(name) for name in normal_data.columns])
#     print(dag)
#     print(dag.edges)
#     return dag


def my_grasp(normal_data):
    G = grasp(normal_data)
    dag = GeneralGraph_to_DiGraph(G)
    dag = dag_vars_num_to_name(dag, [str(name) for name in normal_data.columns])
    print('GRaSP')
    print(dag)
    print(dag.edges)
    return dag


def get_other_methods_dag(normal_data):
    pc_dag = my_pc(normal_data)
    ges_dag = my_ges(normal_data)
    grasp_dag = my_grasp(normal_data)
    ICALiNGAM_dag = my_ICALiNGAM(normal_data, 0.05)
    # GIN_dag = my_GIN(normal_data)

    other_methods_dag = {'pc': pc_dag,
                         'ges': ges_dag,
                         'grasp': grasp_dag,
                         'ICALiNGAM': ICALiNGAM_dag}
    return other_methods_dag


def exp(output_path, true_dag, given_dag, target_node, normal_data, anomalous_data, labels, dag_score_dict,
        causal_mechanism_reusable_setter, ops_add, ops_remove, ops_reverse, cause_node_only,
        num_distribution_samples, pruning, llm_dag, independence_tester,
        iterationLimit=None, timeLimit=None):
    create_directory_if_not_exists(output_path)
    initialState = State(given_dag, target_node, normal_data, anomalous_data, labels, true_dag=true_dag, dag=None,
                         dag_score_dict=dag_score_dict,
                         causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                         ops_add=ops_add, ops_remove=ops_remove, ops_reverse=ops_reverse,
                         cause_node_only=cause_node_only,
                         num_distribution_samples=num_distribution_samples, pruning=pruning, llm_dag=llm_dag,
                         independence_tester=independence_tester,
                         )
    dag_score_dict = initialState.dag_score_dict
    dag_record_list = initialState.dag_record_list

    given_dag_score = calculate_dag_score(dag=given_dag, target_node=target_node, normal_data=normal_data,
                                          anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                                          dag_record_list=dag_record_list,
                                          causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                                          true_dag=true_dag, given_dag=given_dag,
                                          num_distribution_samples=num_distribution_samples)
    print('given_dag_score:', given_dag_score)

    if iterationLimit:
        searcher = mcts(iterationLimit=iterationLimit, rolloutPolicy=rewardPolicy, pruning=pruning, explorationConstant=0.05)
    if timeLimit:
        searcher = mcts(timeLimit=timeLimit, rolloutPolicy=rewardPolicy, pruning=pruning, explorationConstant=0.05)
    t0 = time.time()
    searcher.search(initialState=initialState)
    print('total time:', time.time() - t0)
    print('roundNums:', searcher.roundNums)

    true_dag_score = calculate_dag_score(dag=true_dag, target_node=target_node, normal_data=normal_data,
                                         anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
                                         dag_record_list=dag_record_list,
                                         causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
                                         true_dag=true_dag, given_dag=given_dag,
                                         num_distribution_samples=num_distribution_samples)
    dag_record_list[-1]['index'] = 'true'
    print('true_dag_score:', true_dag_score)

    # resultList = searcher.getResultList()
    # i = 0
    # for dag in resultList:
    #     calculate_dag_score(dag=dag, target_node=target_node, normal_data=normal_data,
    #                         anomalous_data=anomalous_data, labels=labels, dag_score_dict=dag_score_dict,
    #                         dag_record_list=dag_record_list,
    #                         causal_mechanism_reusable_setter=causal_mechanism_reusable_setter,
    #                         true_dag=true_dag, given_dag=given_dag,
    #                         num_distribution_samples=num_distribution_samples)
    #     dag_record_list[-1]['index'] = 'result:'+str(i)
    #     i += 1

    save_result(dag_record_list, dag_score_dict, output_path=output_path)


if __name__ == '__main__':
    pass
