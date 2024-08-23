import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dowhy import gcm
from scipy.stats import norm

from mcts.calculate_utils import CausalMechanismReusableSetter, attribute_anomalies_multiple
from mcts.graph_utils import get_ordered_node_list


def generate_data_rate(dag, data, target_node, abnormal_rate_low, abnormal_rate_high, normal_data_size=None, anomalous_data_size=None,
                  num_distribution_samples=3000):
    data = data.sort_values(by=target_node)

    total_rows = len(data)
    low_rows = int(total_rows * abnormal_rate_low)
    high_rows = int(total_rows * abnormal_rate_high)
    low_df = data.head(low_rows)
    high_df = data.tail(high_rows)
    middle_df = data.iloc[low_rows:-high_rows]

    normal_data = middle_df
    anomalous_data = pd.concat([low_df, high_df])

    if (normal_data_size is not None) and len(normal_data) > normal_data_size:
        normal_data = normal_data.sample(normal_data_size)
    if (anomalous_data_size is not None) and len(anomalous_data) > anomalous_data_size:
        anomalous_data = anomalous_data.sample(anomalous_data_size)

    causal_model = gcm.StructuralCausalModel(dag)
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(dag))
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
    t0 = time.time()
    print('start:', datetime.fromtimestamp(t0))
    # attribution_scores = gcm.attribute_anomalies(causal_model, target_node, anomaly_samples=anomalous_data,
    #                                              num_distribution_samples=num_distribution_samples)
    attribution_scores = attribute_anomalies_multiple(causal_model, target_node, anomaly_samples=anomalous_data,
                                                      num_distribution_samples=num_distribution_samples)
    t1 = time.time()
    print('end:', datetime.fromtimestamp(t1))
    print('time:', t1-t0)
    column_list = list(dag.nodes)
    prediction = np.zeros(shape=[len(anomalous_data), len(column_list)])
    for name in attribution_scores.keys():
        ci = column_list.index(name)
        ri = 0
        for value in attribution_scores[name]:
            prediction[ri, ci] = value
            ri += 1
    prediction = (prediction == prediction.max(axis=1, keepdims=True)).astype(int)
    labels_data = pd.DataFrame(prediction, columns=column_list)
    return normal_data, anomalous_data, labels_data


def generate_data_threshold(dag, data, target_node, abnormal_threshold_low=-math.inf, abnormal_threshold_high=math.inf, normal_data_size=None, anomalous_data_size=None,
                  num_distribution_samples=3000):

    normal_data = data[(data[target_node] > abnormal_threshold_low) & (data[target_node] < abnormal_threshold_high)]
    anomalous_data = data[(data[target_node] <= abnormal_threshold_low) | (data[target_node] >= abnormal_threshold_high)]

    if (normal_data_size is not None) and len(normal_data) > normal_data_size:
        normal_data = normal_data.sample(normal_data_size)
    if (anomalous_data_size is not None) and len(anomalous_data) > anomalous_data_size:
        anomalous_data = anomalous_data.sample(anomalous_data_size)

    causal_model = gcm.StructuralCausalModel(dag)
    causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(dag))
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
    t0 = time.time()
    print('start:', datetime.fromtimestamp(t0))
    # attribution_scores = gcm.attribute_anomalies(causal_model, target_node, anomaly_samples=anomalous_data,
    #                                              num_distribution_samples=num_distribution_samples)
    attribution_scores = attribute_anomalies_multiple(causal_model, target_node, anomaly_samples=anomalous_data,
                                                      num_distribution_samples=num_distribution_samples)
    t1 = time.time()
    print('end:', datetime.fromtimestamp(t1))
    print('time:', t1-t0)
    column_list = list(dag.nodes)
    prediction = np.zeros(shape=[len(anomalous_data), len(column_list)])
    for name in attribution_scores.keys():
        ci = column_list.index(name)
        ri = 0
        for value in attribution_scores[name]:
            prediction[ri, ci] = value
            ri += 1
    prediction = (prediction == prediction.max(axis=1, keepdims=True)).astype(int)
    labels_data = pd.DataFrame(prediction, columns=column_list)
    return normal_data, anomalous_data, labels_data