from __future__ import division

import time
import math
import random


# def randomPolicy(state):
#     raise Exception('ERROR: randomPolicy')
#     while not state.isTerminal():
#         try:
#             action = random.choice(state.getPossibleActions())
#         except IndexError:
#             raise Exception("Non-terminal state has no possible actions: " + str(state))
#         state = state.takeAction(action)
#     return state.getReward()


def rewardPolicy(state):
    return state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.bestReward = 0
        self.bestChildrenReward = 0
        self.patience = 0
        self.reward = None

        self.onceChildrenBetter = False
        self.patience_upper_limit = 3

    def __str__(self):
        s = []
        s.append("totalReward: %s" % (self.totalReward))
        s.append("numVisits: %d" % (self.numVisits))
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1,
                 rolloutPolicy=None, pruning=False):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.roundNums = 0

        self.allNodes = []
        self.selectNodeBestRewardFactor = 0.9
        self.bestNodeSelfRewardFactor = 0.9
        self.pruning = pruning

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        self.allNodes.append(self.root)
        reward = self.rollout(self.root.state)
        self.backpropogate(self.root, reward)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        # bestChild = self.getBestChild(self.root, 0)
        # action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        # if needDetails:
        #     return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        # else:
        #     return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        self.roundNums += 1
        print('executeRound:', self.roundNums)
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)
        print('reward:', reward)

    def selectNode(self, node):
        consideredNodes = []
        for node in self.allNodes:
            if not self.isFullyExpanded(node):
                consideredNodes.append(node)
        if not consideredNodes:
            return node
        node = self.getBestNode(consideredNodes, self.explorationConstant)
        if (not self.pruning) or self.needExpend(node):
            return self.expand(node)
        else:
            return node

    def expand(self, node):
        action = node.state.getNextPossibleAction()
        if action is None:
            return node
        else:
            node.patience = 0
            newNode = treeNode(node.state.takeAction(action), node)
            node.children[action] = newNode
            self.allNodes.append(newNode)
            return newNode

        print("ERROR:Should never reach here")
        raise Exception("Should never reach here")
        # return node

    def backpropogate(self, node, reward):
        node.numVisits += 1
        node.totalReward += reward
        node.reward = reward
        if reward > node.bestReward:
            node.bestReward = reward
            node.patience = 0
        else:
            node.patience += 1
        if node.parent is not None and node.reward > node.parent.bestChildrenReward:
            node.parent.bestChildrenReward = node.reward
        node = node.parent
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            if reward > node.bestReward:
                node.onceChildrenBetter = True
                node.bestReward = reward
                node.patience = 0
            else:
                node.patience += 1
            if node.parent is not None and node.reward > node.parent.bestChildrenReward:
                node.parent.bestChildrenReward = node.reward
            node = node.parent

    # def getBestChild(self, node, explorationValue):
    #     # if len(node.children) == 0:
    #     #     return node
    #     bestValue = float("-inf")
    #     bestNodes = []
    #     if node.numVisits >= len(node.children) * 3:
    #         selectNodeBestRewardFactor = self.selectNodeBestRewardFactor
    #     else:
    #         selectNodeBestRewardFactor = 1
    #     for child in node.children.values():
    #         nodeValue = selectNodeBestRewardFactor * child.bestReward \
    #                     + (1 - selectNodeBestRewardFactor) * (child.totalReward / child.numVisits) \
    #                     + explorationValue * math.sqrt(math.log(max(node.numVisits, 1)) / max(child.numVisits, 1))
    #         # nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
    #         #     2 * math.log(node.numVisits) / child.numVisits)
    #         if nodeValue > bestValue:
    #             bestValue = nodeValue
    #             bestNodes = [child]
    #         elif nodeValue == bestValue:
    #             bestNodes.append(child)
    #     return random.choice(bestNodes)

    def getBestNode(self, consideredNodes, explorationValue):
        # raise NotImplementedError()
        bestValue = float("-inf")
        bestNodes = []
        for node in consideredNodes:
            if node.numVisits >= 3:
                bestNodeSelfRewardFactor = self.bestNodeSelfRewardFactor
            else:
                bestNodeSelfRewardFactor = 1
            if node.parent:
                parent_numVisits = node.parent.numVisits
            else:
                parent_numVisits = max(node.numVisits, 1)
            nodeValue = bestNodeSelfRewardFactor * node.reward \
                        + (1 - bestNodeSelfRewardFactor) * node.bestChildrenReward \
                        + explorationValue * math.sqrt((math.log(parent_numVisits) + 1) / max(node.numVisits, 1))
            # nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
            #     2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [node]
            elif nodeValue == bestValue:
                bestNodes.append(node)
        return random.choice(bestNodes)

    # def isTerminal(self, node):
    #     if len(node.children) == 0 and node.state.isFullyExpanded():
    #         return True

    def isFullyExpanded(self, node):
        if node.state.isFullyExpanded():
            return True
        else:
            if len(node.children) > 10 and node.onceChildrenBetter is True:
                return True
        return False

    def needExpend(self, node):
        if len(node.children) < 3:
            return True
        elif len(node.children) < 10 and node.onceChildrenBetter is False:
            return True
        else:
            if node.patience >= node.patience_upper_limit:
                node.patience_upper_limit += 3
                return True
            else:
                return False
