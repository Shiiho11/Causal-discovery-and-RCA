import networkx as nx
import numpy as np
from sklearn.utils import check_array


def has_cycle(dag: nx.DiGraph):
    cycles = nx.recursive_simple_cycles(dag)
    if cycles:
        return True
    else:
        return False


def dag_vars_num_to_name(dag, node_names):
    new_dag = nx.DiGraph()
    new_dag.add_nodes_from([node_names[i] for i in dag.nodes])
    new_dag.add_edges_from([(node_names[i], node_names[j]) for (i, j) in dag.edges])
    return new_dag


def adjacency_matrix_to_dag(adjacency_matrix_, node_names, lower_limit=0.01):
    dag = nx.DiGraph()
    dag.add_nodes_from(node_names)
    # edges
    B = check_array(np.nan_to_num(adjacency_matrix_))
    labels = [f'{col}' for i, col in enumerate(node_names)]
    names = labels if labels else [f'x{i}' for i in range(len(B))]
    idx = np.abs(B) > lower_limit
    dirs = np.where(idx)
    for to, from_, coef in zip(dirs[0], dirs[1], B[idx]):
        dag.add_edge(names[from_], names[to], weight=coef)
    return dag


def GeneralGraph_to_DiGraph(gg):
    dag = nx.DiGraph()
    # 遍历 GeneralGraph 中的节点，并添加到 networkx 图中
    nodes = gg.get_nodes()
    dag.add_nodes_from(range(len(nodes)))
    # 遍历 GeneralGraph 中的边，并添加到 networkx 图中
    for edge in gg.get_graph_edges():
        source = nodes.index(edge.get_node1())
        target = nodes.index(edge.get_node2())
        dag.add_edge(source, target)
    return dag


def get_ordered_node_list(dag):
    node_list = list(dag.nodes)
    node_list.sort()
    return node_list


def dag_similarity(true_dag, dag):
    if len(dag.nodes) <= 1:
        return 0
    score = 0
    for source in dag.nodes:
        for target in dag.nodes:
            if source != target:
                given = 0
                current = 0
                if (source, target) in dag.edges:
                    current = 1
                if (source, target) in true_dag.edges:
                    given = 1
                if given == current:
                    score += 1
    n = len(dag.nodes)
    score = score / (n ** 2 - n)
    return score


def normalized_hamming_distance(true_dag, dag):
    if len(dag.nodes) <= 1:
        return None
    distance = 0
    for source in dag.nodes:
        for target in dag.nodes:
            if source != target:
                true = 0
                given = 0
                if (source, target) in true_dag.edges:
                    true = 1
                if (source, target) in dag.edges:
                    given = 1
                if true != given:
                    distance += 1
    n = len(dag.nodes)
    NHD = distance / (n ** 2 - n)
    return NHD


def get_adjacency_matrix_str(dag, node_list):
    # 获取邻接矩阵 按照初始dag的node顺序
    nodes = node_list
    num = len(nodes)
    adjacency_matrix = [[0] * num for _ in range(num)]
    for i in range(num):
        for j in range(num):
            if (nodes[i], nodes[j]) in dag.edges:
                adjacency_matrix[i][j] = 1
    # 邻接矩阵转换成字符串 唯一表示
    key_string = '\n'.join([''.join(['0' if num == 0 else '1' for num in row]) for row in adjacency_matrix])
    return key_string


def get_condition_vars(dag, u, v):
    # 节点不相连
    g = dag.to_undirected()
    if not nx.has_path(g, u, v):
        return None
    # v到u有path, 错误
    if nx.has_path(dag, v, u):
        return None
    pa = list(dag.predecessors(v))
    # u是v的pa, 错误
    if u in pa:
        return None
    if not pa:
        return None
    return pa


def graph_evaluate(true_dag, dag):
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for source in dag.nodes:
        for target in dag.nodes:
            if source != target:
                true = 0
                given = 0
                if (source, target) in true_dag.edges:
                    true = 1
                if (source, target) in dag.edges:
                    given = 1
                if true == 0 and given == 0:
                    tn += 1
                elif true == 1 and given == 1:
                    tp += 1
                elif true == 0 and given == 1:
                    fp += 1
                elif true == 1 and given == 0:
                    fn += 1
    if (tp + fp + tn + fn) == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if (precision + recall) == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, F1
