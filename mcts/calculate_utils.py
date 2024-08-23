import json
import time
from copy import copy

import networkx as nx
import dowhy.gcm as gcm
import numpy as np
import pandas as pd
from scipy.stats import norm
from dowhy.gcm.causal_models import (
    PARENTS_DURING_FIT,
)
from dowhy.graph import get_ordered_predecessors
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

from mcts.graph_utils import get_ordered_node_list, get_adjacency_matrix_str, dag_similarity, \
    normalized_hamming_distance, graph_evaluate


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 减去每行最大的值
    e_x = np.exp(x)
    f_x = e_x / np.sum(e_x, axis=1, keepdims=True)
    return f_x


class CausalMechanismReusableSetter:
    def __init__(self, nodes):
        self.nodes = nodes
        self.causal_mechanism_repository = dict()

    def set_causal_mechanism_and_fit(self, causal_model, node, data, default_causal_mechanism):
        if node not in self.causal_mechanism_repository:
            self.causal_mechanism_repository[node] = dict()
        dag = causal_model.graph
        parents = list(dag.predecessors(node))
        parents = tuple([node for node in self.nodes if node in parents])
        if parents in self.causal_mechanism_repository[node]:
            causal_model.set_causal_mechanism(node, self.causal_mechanism_repository[node][parents])
            # 为了通过验证, 添加PARENTS_DURING_FIT信息
            target_node = node
            causal_model.graph.nodes[target_node][PARENTS_DURING_FIT] = get_ordered_predecessors(
                causal_model.graph, target_node)
        else:
            causal_mechanism = default_causal_mechanism
            causal_model.set_causal_mechanism(node, causal_mechanism)
            gcm.fitting_sampling.fit_causal_model_of_target(causal_model, node, data)
            self.causal_mechanism_repository[node][parents] = causal_mechanism
            # self.causal_mechanism_repository[node][parents] = causal_model.causal_mechanism(node)


class IndependenceTester:
    def __init__(self, data, var_list):
        self.data = data
        self.var_list = var_list
        self.p_value_of_triple = dict()

    def independence_test(self, X, Y, conditioned_on=None, method='kernel'):
        t_s = time.time()
        triple = []
        for var in self.var_list:
            if var in (X, Y):
                triple.append(var)
        if conditioned_on:
            for var in self.var_list:
                if var in conditioned_on:
                    triple.append(var)
        triple = tuple(triple)
        if triple in self.p_value_of_triple:
            return self.p_value_of_triple[triple]
        else:
            if conditioned_on:
                p_value = gcm.independence_test(self.data[X].values, self.data[Y].values,
                                                conditioned_on=self.data[conditioned_on].values, method=method)
            else:
                p_value = gcm.independence_test(self.data[X].values, self.data[Y].values, conditioned_on=None,
                                                method=method)
            self.p_value_of_triple[triple] = p_value
            print('time_independence_test:', time.time() - t_s)
            return p_value


# TODO
def attribute_anomalies_multiple(causal_model, target_node, anomaly_samples, num_distribution_samples=1500, test_num=3):
    attribution_scores_list = []
    for i in range(test_num):
        attribution_scores_list.append(
            gcm.attribute_anomalies(causal_model, target_node, anomaly_samples=anomaly_samples,
                                    num_distribution_samples=num_distribution_samples))
    for name in attribution_scores_list[0].keys():
        for i in range(1, test_num):
            attribution_scores_list[0][name] += attribution_scores_list[i][name]
        attribution_scores_list[0][name] /= test_num
    attribution_scores = attribution_scores_list[0]
    # print(attribution_scores)
    return attribution_scores


def calculate_dag_score(dag, target_node, normal_data, anomalous_data, labels, dag_score_dict: dict,
                        dag_record_list: list, causal_mechanism_reusable_setter=None,
                        true_dag=nx.DiGraph(), given_dag=nx.DiGraph(), num_distribution_samples=1500, test_num=3):
    node_list = get_ordered_node_list(true_dag)
    key_string = get_adjacency_matrix_str(dag, node_list)
    if key_string in dag_score_dict.keys():
        dag_record_list.append(copy(dag_score_dict[key_string]))
        dag_record_list[-1]['index'] = len(dag_record_list) - 1
        return dag_score_dict[key_string]['score']

    causal_model = gcm.StructuralCausalModel(dag)
    t_cm_s = time.time()
    if causal_mechanism_reusable_setter is None:
        gcm.auto.assign_causal_mechanisms(causal_model, normal_data)
        # for node in dag.nodes:
        #     if len(list(dag.predecessors(node))) > 0:
        #         causal_model.set_causal_mechanism(node,
        #                                           gcm.AdditiveNoiseModel(
        #                                               prediction_model=gcm.ml.create_linear_regressor(),
        #                                               noise_model=gcm.ScipyDistribution(norm)))
        #     else:
        #         causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(norm))
        gcm.fit(causal_model, normal_data)
    else:
        for node in dag.nodes:
            if len(list(dag.predecessors(node))) > 0:
                causal_mechanism_reusable_setter.set_causal_mechanism_and_fit(causal_model, node, normal_data,
                                                                              gcm.AdditiveNoiseModel(
                                                                                  prediction_model=gcm.ml.create_linear_regressor(),
                                                                                  noise_model=gcm.ScipyDistribution(
                                                                                      norm)))
            else:
                causal_mechanism_reusable_setter.set_causal_mechanism_and_fit(causal_model, node, normal_data,
                                                                              gcm.ScipyDistribution(norm))
    print('time_set_causal_mechanism:', time.time() - t_cm_s)
    t_as_s = time.time()
    attribution_scores = attribute_anomalies_multiple(causal_model, target_node, anomaly_samples=anomalous_data,
                                                      num_distribution_samples=num_distribution_samples,
                                                      test_num=test_num)
    # print(attribution_scores)
    print('time_attribution_scores:', time.time() - t_as_s)

    # 计算分数
    # 预测数据numpy
    column_list = list(labels.columns)
    prediction = np.zeros(shape=[len(labels), len(column_list)])
    for name in attribution_scores.keys():
        if name in column_list:
            ci = column_list.index(name)
            ri = 0
            for value in attribution_scores[name]:
                prediction[ri, ci] = value
                ri += 1
    # softmax
    prediction = softmax(prediction)
    # 标签numpy
    true_labels = labels.values
    # Cross Entropy Loss
    loss = log_loss(true_labels, prediction)
    # 1d
    prediction_1d = prediction.argmax(axis=1)
    true_labels_1d = true_labels.argmax(axis=1)

    # RCA 评估指标
    RCA_accuracy = accuracy_score(true_labels_1d, prediction_1d)
    RCA_precision = precision_score(true_labels_1d, prediction_1d, average='macro')
    RCA_recall = recall_score(true_labels_1d, prediction_1d, average='macro')
    RCA_F1 = f1_score(true_labels_1d, prediction_1d, average='macro')
    RCA_precision_micro = precision_score(true_labels_1d, prediction_1d, average='micro')
    RCA_recall_micro = recall_score(true_labels_1d, prediction_1d, average='micro')
    RCA_F1_micro = f1_score(true_labels_1d, prediction_1d, average='micro')
    # graph_precision, graph_recall, graph_F1
    graph_accuracy, graph_precision, graph_recall, graph_F1 = graph_evaluate(true_dag, dag)
    # Normalized Hamming Distance (NHD)
    NHD_true = normalized_hamming_distance(true_dag, dag)
    NHD_given = normalized_hamming_distance(given_dag, dag)
    empty_dag = nx.DiGraph()
    empty_dag.add_nodes_from(list(dag.nodes))
    NHD_empty = normalized_hamming_distance(empty_dag, dag)

    score = RCA_F1 - NHD_empty * 0.2
    dag_score_dict[key_string] = {'index': len(dag_score_dict), 'score': score, 'RCA_loss': loss,
                                  'RCA_accuracy': RCA_accuracy,
                                  'RCA_precision': RCA_precision, 'RCA_recall': RCA_recall, 'RCA_F1': RCA_F1,
                                  'graph_accuracy': graph_accuracy,
                                  'graph_precision': graph_precision, 'graph_recall': graph_recall,
                                  'graph_F1': graph_F1,
                                  'NHD_true': NHD_true, 'NHD_given': NHD_given, 'NHD_empty': NHD_empty,
                                  'RCA_precision_micro': RCA_precision_micro, 'RCA_recall_micro': RCA_recall_micro,
                                  'RCA_F1_micro': RCA_F1_micro,
                                  'nodes': list(dag.nodes), 'edges': list(dag.edges)}
    dag_record_list.append(dag_score_dict[key_string])
    dag_record_list[-1]['index'] = len(dag_record_list) - 1
    return score


def save_result(dag_record_list, dag_score_dict, output_path='./'):
    with open(output_path + 'record.json', 'w', encoding='utf-8') as f:
        json.dump(dag_record_list, f)
    dag_record_csv_data = []
    for dag_info in dag_record_list:
        dag_record_csv_data.append(
            [dag_info['index'], dag_info['score'], dag_info['RCA_loss'], dag_info['RCA_accuracy'],
             dag_info['RCA_precision'], dag_info['RCA_recall'], dag_info['RCA_F1'], dag_info['graph_accuracy'],
             dag_info['graph_precision'], dag_info['graph_recall'], dag_info['graph_F1'],
             dag_info['NHD_true'], dag_info['NHD_given'], dag_info['NHD_empty'],
             dag_info['RCA_precision_micro'], dag_info['RCA_recall_micro'], dag_info['RCA_F1_micro'],
             dag_info['nodes'], dag_info['edges']])
    pd.DataFrame(dag_record_csv_data,
                 columns=['index', 'score', 'RCA_loss', 'RCA_accuracy', 'RCA_precision', 'RCA_recall', 'RCA_F1',
                          'graph_accuracy', 'graph_precision', 'graph_recall', 'graph_F1', 'NHD_true', 'NHD_given',
                          'NHD_empty',
                          'RCA_precision_micro', 'RCA_recall_micro', 'RCA_F1_micro',
                          'nodes', 'edges'
                          ]).to_csv(output_path + 'record.csv', index=False)

    dag_score_list = list(dag_score_dict.values())
    with open(output_path + 'score.json', 'w', encoding='utf-8') as f:
        json.dump(dag_score_list, f)
    dag_score_csv_data = []
    for dag_info in dag_score_list:
        dag_score_csv_data.append(
            [dag_info['index'], dag_info['score'], dag_info['RCA_loss'], dag_info['RCA_accuracy'],
             dag_info['RCA_precision'], dag_info['RCA_recall'], dag_info['RCA_F1'], dag_info['graph_accuracy'],
             dag_info['graph_precision'], dag_info['graph_recall'], dag_info['graph_F1'],
             dag_info['NHD_true'], dag_info['NHD_given'], dag_info['NHD_empty'],
             dag_info['RCA_precision_micro'], dag_info['RCA_recall_micro'], dag_info['RCA_F1_micro'],
             dag_info['nodes'], dag_info['edges']])
    pd.DataFrame(dag_score_csv_data,
                 columns=['index', 'score', 'RCA_loss', 'RCA_accuracy', 'RCA_precision', 'RCA_recall', 'RCA_F1',
                          'graph_accuracy', 'graph_precision', 'graph_recall', 'graph_F1', 'NHD_true', 'NHD_given',
                          'NHD_empty',
                          'RCA_precision_micro', 'RCA_recall_micro', 'RCA_F1_micro',
                          'nodes', 'edges'
                          ]).to_csv(output_path + 'score.csv', index=False)


def save_other_methods_result(dag_record_list, output_path='./'):
    with open(output_path + 'other_methods.json', 'w', encoding='utf-8') as f:
        json.dump(dag_record_list, f)
    dag_record_csv_data = []
    for dag_info in dag_record_list:
        dag_record_csv_data.append(
            [dag_info['index'], dag_info['score'], dag_info['RCA_loss'], dag_info['RCA_accuracy'],
             dag_info['RCA_precision'], dag_info['RCA_recall'], dag_info['RCA_F1'], dag_info['graph_accuracy'],
             dag_info['graph_precision'], dag_info['graph_recall'], dag_info['graph_F1'],
             dag_info['NHD_true'], dag_info['NHD_given'], dag_info['NHD_empty'],
             dag_info['RCA_precision_micro'], dag_info['RCA_recall_micro'], dag_info['RCA_F1_micro'],
             dag_info['nodes'], dag_info['edges']])
    pd.DataFrame(dag_record_csv_data,
                 columns=['index', 'score', 'RCA_loss', 'RCA_accuracy', 'RCA_precision', 'RCA_recall', 'RCA_F1',
                          'graph_accuracy', 'graph_precision', 'graph_recall', 'graph_F1', 'NHD_true', 'NHD_given',
                          'NHD_empty',
                          'RCA_precision_micro', 'RCA_recall_micro', 'RCA_F1_micro',
                          'nodes', 'edges'
                          ]).to_csv(output_path + 'other_methods.csv', index=False)
