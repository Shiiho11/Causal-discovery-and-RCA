from __future__ import division

import random
from copy import copy

import networkx as nx

from mcts.graph_utils import has_cycle, get_ordered_node_list, get_adjacency_matrix_str, get_condition_vars
from mcts.calculate_utils import CausalMechanismReusableSetter, calculate_dag_score
from bcgllm.llm_utils import get_possible_actions_with_llm


def actionsInfo_to_PossibleAction(actionsInfo):
    ops = actionsInfo['ops']
    u = actionsInfo['u']
    v = actionsInfo['v']
    return Action(ops, u, v)


class State:
    def __init__(self, given_dag, target_node, normal_data, anomalous_data, labels, true_dag=None, dag=None,
                 dag_score_dict=None, causal_mechanism_reusable_setter=None,
                 ops_add=True, ops_remove=True, ops_reverse=True, cause_node_only=False, num_distribution_samples=1500,
                 pruning=False, llm_dag=None, independence_tester=None, possible_actions_num_limit=None):
        # 只在初始State时使用
        # 不变
        self.given_dag = given_dag
        self.target_node = target_node
        self.normal_data = normal_data
        self.anomalous_data = anomalous_data
        self.labels = labels
        self.true_dag = true_dag
        if self.true_dag is None:
            self.true_dag = self.given_dag

        self.dag_score_dict = dag_score_dict
        if self.dag_score_dict is None:
            self.dag_score_dict = dict()
        self.dag_record_list = []
        self.causal_mechanism_reusable_setter = causal_mechanism_reusable_setter
        if self.causal_mechanism_reusable_setter is None:
            self.causal_mechanism_reusable_setter = CausalMechanismReusableSetter(get_ordered_node_list(self.given_dag))

        # 边变化选项
        self.ops_add = ops_add
        self.ops_remove = ops_remove
        self.ops_reverse = ops_reverse
        self.cause_node_only = cause_node_only

        self.num_distribution_samples = num_distribution_samples

        # 剪枝选项
        self.pruning = pruning
        self.independence_tester = independence_tester
        self.possible_actions_num_limit = possible_actions_num_limit
        self.llm_dag = llm_dag

        # 每个state不同, 需要改变
        self.dag = dag
        if self.dag is None:
            self.dag = self.given_dag
        self.possibleActions = None
        self.priority_possibleActionsInfo = None
        self.normal_priority_possibleActionsInfo = None
        self.others_possibleActionsInfo = None

        # test
        self.dag_set = set()

    def initPossibleActions(self):
        if self.possibleActions is not None:
            return

        # 所有可能的选择
        possibleActionsInfo = self._getAllPossibleActionsInfo()

        if not self.pruning:
            possibleActions = []
            for actionsInfo in possibleActionsInfo:
                possibleActions.append(actionsInfo_to_PossibleAction(actionsInfo))
            # random.shuffle(possibleActions)
            self.possibleActions = possibleActions
            return

        # pruning
        # independence_test
        possibleActionsInfo = self._independence_tester_for_possible_actions(possibleActionsInfo)
        # delete
        possibleActionsInfo = self._delete_possible_actions(possibleActionsInfo)
        # 优选
        priority_possibleActionsInfo, others_possibleActionsInfo = self._priority_possible_actions(possibleActionsInfo)
        # 一般 剩下的是差的选择
        normal_priority_possibleActionsInfo, others_possibleActionsInfo = self._normal_priority_possible_actions(
            others_possibleActionsInfo)

        random.shuffle(priority_possibleActionsInfo)
        random.shuffle(normal_priority_possibleActionsInfo)
        random.shuffle(others_possibleActionsInfo)

        self.priority_possibleActionsInfo = priority_possibleActionsInfo
        self.normal_priority_possibleActionsInfo = normal_priority_possibleActionsInfo
        self.others_possibleActionsInfo = others_possibleActionsInfo
        self.possibleActions = True

    def getNextPossibleAction(self):
        self.initPossibleActions()
        # self._check_dag_set()
        # print('getNextPossibleAction')
        # print(self.dag.edges)
        # print(self.possibleActions)
        # print(self.priority_possibleActionsInfo)
        # print(self.normal_priority_possibleActionsInfo)
        # print(self.others_possibleActionsInfo)
        if not self.pruning:
            if self.possibleActions:
                action = self.possibleActions.pop()
                # print(action)
                return action
            else:
                return None

        if self.priority_possibleActionsInfo:
            actionsInfo = self.priority_possibleActionsInfo.pop()
        elif self.normal_priority_possibleActionsInfo:
            actionsInfo = self.normal_priority_possibleActionsInfo.pop()
        elif self.others_possibleActionsInfo:
            actionsInfo = self.others_possibleActionsInfo.pop()
        else:
            return None
            # raise Exception('no PossibleAction')
        action = actionsInfo_to_PossibleAction(actionsInfo)
        # print(action)
        return action

    def takeAction(self, action):
        # print('takeAction')
        # print(self.dag.edges)
        # print(action)
        newState = copy(self)
        newState.possibleActions = None
        newState.priority_possibleActionsInfo = None
        newState.normal_priority_possibleActionsInfo = None
        newState.others_possibleActionsInfo = None
        new_dag = self.dag.copy()
        if action.ops == 'add':
            new_dag.add_edge(action.u, action.v)
        elif action.ops == 'remove':
            new_dag.remove_edge(action.u, action.v)
        elif action.ops == 'reverse':
            new_dag.remove_edge(action.u, action.v)
            new_dag.add_edge(action.v, action.u)
        else:
            raise Exception('ERROR: unknown ops')
        newState.dag = new_dag
        newState.dag_set.add(get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)))
        return newState

    def getReward(self):
        self.score = calculate_dag_score(dag=self.dag, target_node=self.target_node, normal_data=self.normal_data,
                                         anomalous_data=self.anomalous_data, labels=self.labels,
                                         dag_score_dict=self.dag_score_dict,
                                         dag_record_list=self.dag_record_list,
                                         causal_mechanism_reusable_setter=self.causal_mechanism_reusable_setter,
                                         true_dag=self.true_dag, given_dag=self.given_dag,
                                         num_distribution_samples=self.num_distribution_samples)
        return self.score

    def isFullyExpanded(self):
        self.initPossibleActions()
        # self._check_dag_set()
        if not self.pruning:
            if len(self.possibleActions) == 0:
                return True
            else:
                return False

        temp = len(self.priority_possibleActionsInfo) + len(self.normal_priority_possibleActionsInfo) + len(
            self.others_possibleActionsInfo)
        if temp == 0:
            return True
        else:
            return False

    # def isTerminal(self):
    #     pass

    def _getAllPossibleActionsInfo(self):
        possibleActionsInfo = []
        node_list = get_ordered_node_list(self.dag)
        for u in node_list:
            for v in node_list:
                if u == v:
                    continue
                if (u, v) in list(self.dag.edges):
                    # remove
                    if self.ops_remove:
                        new_dag = self.dag.copy()
                        new_dag.remove_edge(u, v)
                        if get_adjacency_matrix_str(new_dag, node_list) not in self.dag_set:
                            possibleActionsInfo.append({'ops': 'remove', 'u': u, 'v': v})
                    # reverse
                    if self.ops_reverse:
                        new_dag = self.dag.copy()
                        new_dag.remove_edge(u, v)
                        new_dag.add_edge(v, u)
                        if (not has_cycle(new_dag)) and (
                                get_adjacency_matrix_str(new_dag, node_list) not in self.dag_set):
                            possibleActionsInfo.append({'ops': 'reverse', 'u': u, 'v': v})
                else:
                    # add
                    if self.ops_add:
                        if (not self.cause_node_only) or nx.has_path(self.dag, v, self.target_node):
                            new_dag = self.dag.copy()
                            new_dag.add_edge(u, v)
                            if (not has_cycle(new_dag)) and (
                                    get_adjacency_matrix_str(new_dag, node_list) not in self.dag_set):
                                possibleActionsInfo.append({'ops': 'add', 'u': u, 'v': v})
        return possibleActionsInfo

    def _independence_tester_for_possible_actions(self, possibleActionsInfo):
        new_possibleActionsInfo = []
        for actionsInfo in possibleActionsInfo:
            ops = actionsInfo['ops']
            u = actionsInfo['u']
            v = actionsInfo['v']
            if ops == 'add':
                condition_vars = get_condition_vars(self.dag, u, v)
                p_value = self.independence_tester.independence_test(u, v, conditioned_on=condition_vars)
                # if p_value > 0.8:
                #     continue
                new_possibleActionsInfo.append({'ops': ops, 'u': u, 'v': v,
                                                'condition_vars': condition_vars, 'p_value': p_value})
            elif ops == 'remove':
                new_dag = self.dag.copy()
                new_dag.remove_edge(u, v)
                condition_vars = get_condition_vars(new_dag, u, v)
                p_value = self.independence_tester.independence_test(u, v, conditioned_on=condition_vars)
                # if p_value < 0.02:
                #     continue
                new_possibleActionsInfo.append({'ops': ops, 'u': u, 'v': v,
                                                'condition_vars': condition_vars, 'p_value': p_value})
            elif ops == 'reverse':
                new_dag = self.dag.copy()
                new_dag.remove_edge(u, v)
                new_dag.add_edge(v, u)
                # condition_vars_before = get_condition_vars(self.dag, u, v)
                # p_value_before = self.independence_tester.independence_test(u, v, conditioned_on=condition_vars_before)
                condition_vars_after = get_condition_vars(new_dag, u, v)
                p_value_after = self.independence_tester.independence_test(u, v, conditioned_on=condition_vars_after)
                # if p_value_after > 0.8:
                #     continue
                new_possibleActionsInfo.append({'ops': ops, 'u': u, 'v': v,
                                                # 'condition_vars_before': condition_vars_before, 'p_value_before': p_value_before,
                                                'condition_vars_after': condition_vars_after,
                                                'p_value_after': p_value_after})
            else:
                print('ERROR: unknown operation')
        return new_possibleActionsInfo

    def _delete_possible_actions(self, possibleActionsInfo):
        new_possibleActionsInfo = []
        llm_edges = list(self.llm_dag.edges)
        for actionsInfo in possibleActionsInfo:
            ops = actionsInfo['ops']
            u = actionsInfo['u']
            v = actionsInfo['v']
            if ops == 'add':
                p_value = actionsInfo['p_value']
                if p_value > 0.5:
                    continue
                if p_value > 0.1 and (u, v) not in llm_edges:
                    continue
                new_possibleActionsInfo.append(actionsInfo)
            elif ops == 'remove':
                p_value = actionsInfo['p_value']
                if p_value < 0.01:
                    continue
                if p_value < 0.25 and (u, v) in llm_edges:
                    continue
                new_possibleActionsInfo.append(actionsInfo)
            elif ops == 'reverse':
                p_value_after = actionsInfo['p_value_after']
                if p_value_after > 0.5:
                    continue
                if p_value_after > 0.1 and (u, v) not in llm_edges:
                    continue
                new_possibleActionsInfo.append(actionsInfo)
            else:
                print('ERROR: unknown operation')
        return new_possibleActionsInfo

    def _priority_possible_actions(self, possibleActionsInfo):
        priority_possibleActionsInfo = []
        others_possibleActionsInfo = []
        llm_edges = list(self.llm_dag.edges)
        for actionsInfo in possibleActionsInfo:
            ops = actionsInfo['ops']
            u = actionsInfo['u']
            v = actionsInfo['v']
            if ops == 'add':
                p_value = actionsInfo['p_value']
                if p_value < 0.05 and (u, v) in llm_edges:
                    priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            elif ops == 'remove':
                p_value = actionsInfo['p_value']
                if p_value > 0.05 and (u, v) not in llm_edges:
                    priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            elif ops == 'reverse':
                p_value_after = actionsInfo['p_value_after']
                if p_value_after < 0.05 and (u, v) in llm_edges:
                    priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            else:
                print('ERROR: unknown operation')
        return priority_possibleActionsInfo, others_possibleActionsInfo

    def _normal_priority_possible_actions(self, possibleActionsInfo):
        normal_priority_possibleActionsInfo = []
        others_possibleActionsInfo = []
        llm_edges = list(self.llm_dag.edges)
        for actionsInfo in possibleActionsInfo:
            ops = actionsInfo['ops']
            u = actionsInfo['u']
            v = actionsInfo['v']
            if ops == 'add':
                p_value = actionsInfo['p_value']
                if p_value < 0.05 or (u, v) in llm_edges:
                    normal_priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            elif ops == 'remove':
                p_value = actionsInfo['p_value']
                if p_value > 0.05 or (u, v) not in llm_edges:
                    normal_priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            elif ops == 'reverse':
                p_value_after = actionsInfo['p_value_after']
                if p_value_after < 0.05 or (u, v) in llm_edges:
                    normal_priority_possibleActionsInfo.append(actionsInfo)
                else:
                    others_possibleActionsInfo.append(actionsInfo)
            else:
                print('ERROR: unknown operation')
        return normal_priority_possibleActionsInfo, others_possibleActionsInfo

    # def _add_dag_set(self, action):
    #     new_dag = self.dag.copy()
    #     if action.ops == 'add':
    #         new_dag.add_edge(action.u, action.v)
    #     elif action.ops == 'remove':
    #         new_dag.remove_edge(action.u, action.v)
    #     elif action.ops == 'reverse':
    #         new_dag.remove_edge(action.u, action.v)
    #         new_dag.add_edge(action.v, action.u)
    #     else:
    #         print('ERROR: unknown ops')
    #     self.dag_set.add(get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)))

    # def _check_dag_set(self):
    #     if not self.pruning:
    #         for action in self.possibleActions.copy():
    #             new_dag = self.dag.copy()
    #             if action.ops == 'add':
    #                 new_dag.add_edge(action.u, action.v)
    #             elif action.ops == 'remove':
    #                 new_dag.remove_edge(action.u, action.v)
    #             elif action.ops == 'reverse':
    #                 new_dag.remove_edge(action.u, action.v)
    #                 new_dag.add_edge(action.v, action.u)
    #             else:
    #                 print('ERROR: unknown ops')
    #             if get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)) in self.dag_set:
    #                 self.possibleActions.remove(action)
    #         return
    #
    #     for actionsInfo in self.priority_possibleActionsInfo:
    #         action = actionsInfo_to_PossibleAction(actionsInfo)
    #         new_dag = self.dag.copy()
    #         if action.ops == 'add':
    #             new_dag.add_edge(action.u, action.v)
    #         elif action.ops == 'remove':
    #             new_dag.remove_edge(action.u, action.v)
    #         elif action.ops == 'reverse':
    #             new_dag.remove_edge(action.u, action.v)
    #             new_dag.add_edge(action.v, action.u)
    #         else:
    #             print('ERROR: unknown ops')
    #         if get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)) in self.dag_set:
    #             self.priority_possibleActionsInfo.remove(actionsInfo)
    #     for actionsInfo in self.normal_priority_possibleActionsInfo:
    #         action = actionsInfo_to_PossibleAction(actionsInfo)
    #         new_dag = self.dag.copy()
    #         if action.ops == 'add':
    #             new_dag.add_edge(action.u, action.v)
    #         elif action.ops == 'remove':
    #             new_dag.remove_edge(action.u, action.v)
    #         elif action.ops == 'reverse':
    #             new_dag.remove_edge(action.u, action.v)
    #             new_dag.add_edge(action.v, action.u)
    #         else:
    #             print('ERROR: unknown ops')
    #         if get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)) in self.dag_set:
    #             self.normal_priority_possibleActionsInfo.remove(actionsInfo)
    #     for actionsInfo in self.others_possibleActionsInfo:
    #         action = actionsInfo_to_PossibleAction(actionsInfo)
    #         new_dag = self.dag.copy()
    #         if action.ops == 'add':
    #             new_dag.add_edge(action.u, action.v)
    #         elif action.ops == 'remove':
    #             new_dag.remove_edge(action.u, action.v)
    #         elif action.ops == 'reverse':
    #             new_dag.remove_edge(action.u, action.v)
    #             new_dag.add_edge(action.v, action.u)
    #         else:
    #             print('ERROR: unknown ops')
    #         if get_adjacency_matrix_str(new_dag, get_ordered_node_list(self.given_dag)) in self.dag_set:
    #             self.others_possibleActionsInfo.remove(actionsInfo)


class Action:
    def __init__(self, ops, u, v):
        self.ops = ops
        self.u = u
        self.v = v

    def __str__(self):
        return str({'ops': self.ops, 'u': self.u, 'v': self.v})

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.ops == other.ops and self.u == other.u and self.v == other.v

    def __hash__(self):
        return hash((self.ops, self.u, self.v))


if __name__ == "__main__":
    # initialState = NaughtsAndCrossesState()
    # searcher = mcts(timeLimit=1000)
    # action = searcher.search(initialState=initialState)
    #
    # print(action)
    pass
