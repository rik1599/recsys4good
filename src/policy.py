from abc import ABC, abstractmethod
from src.tree import TreeNode
import torch.nn as nn
import pandas as pd
import numpy as np


class Policy(ABC):
    """
    Abstract base class for a selection policy.
    """
    @abstractmethod
    def init(self, **kwargs):
        """
        Initialize the policy.
        """
        pass

    @abstractmethod
    def select(self, nodes: list[TreeNode], n: int, **kwargs) -> set[TreeNode]:
        """
        Select a node from a list of nodes.
        """
        pass

    @abstractmethod
    def estimate(self, node: TreeNode, **kwargs) -> float:
        """
        Estimate the value of a node.
        """
        pass

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the stats for a node after a reward is received.
        """
        pass


class MABTreePolicy(Policy):
    def reset(self):
        self.round_stats = {}


class MABTreeEpsilonGreedyML(MABTreePolicy):
    def __init__(self, model_class, epsilon=0.1, **model_kwargs):
        self.model_class = model_class
        self.model_params = model_kwargs
        self.epsilon = epsilon

    def init(self, **kwargs):
        self.model = self.model_class(**self.model_params)

    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        selectable_nodes = {node: self.estimate(
            node, user=user) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            if np.random.rand() < self.epsilon:
                node = np.random.choice(list(selectable_nodes.keys()))
            else:
                node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(node)
            selectable_nodes.pop(node)

        return selected_nodes
    
    def estimate(self, node, **kwargs):
        user = kwargs['user']
        if node not in self.round_stats and node.is_leaf:
            data = self.model.predict([user], [node.value['missionID']]).item()
            self.round_stats[node] = data

        if node.is_leaf:
            return self.round_stats[node]

        return max(self.estimate(child, user=user) for child in node.children)
    
    def update(self, **kwargs):
        train_df: pd.DataFrame = kwargs['train_df']
        self.model.fit(train_df)

class MABTreeEpsilonGreedy(MABTreePolicy):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def init(self, **kwargs):
        self.average_rewards = pd.Series()

    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        selectable_nodes = {node: self.estimate(
            node, user=user) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            if np.random.rand() < self.epsilon:
                node = np.random.choice(list(selectable_nodes.keys()))
            else:
                node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(node)
            selectable_nodes.pop(node)

        return selected_nodes

    def estimate(self, node, **kwargs):
        user = kwargs['user']

        if node.is_leaf:
            return self.average_rewards.get((user, node.value['missionID']), 0)

        return max(self.estimate(child, user=user) for child in node.children)

    def update(self, **kwargs):
        self.average_rewards = kwargs['train_df'].groupby(
            ['user', 'missionID'])['reward'].mean()


class MABTreeUCB(MABTreePolicy):
    def __init__(self, exploration_rate=1):
        super().__init__()
        self.exploration_rate = exploration_rate
    
    def init(self, **kwargs):
        self.average_rewards = pd.Series()
        self.c = pd.Series()
        self.t = pd.Series()

    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        selectable_nodes = {node: self.estimate(node, user=user) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(node)
            selectable_nodes.pop(node)

        return selected_nodes
    
    def estimate(self, node, **kwargs):
        user = kwargs['user']

        if node.is_leaf:
            mission = node.value['missionID']
            mean = self.average_rewards.get((user, mission), 0)
            count = self.c.get((user, mission), 0) # avoid division by zero
            t = self.t.get(user, 0) # avoid log(0)

            if t == 0 or count == 0:
                return np.inf
            
            return mean + self.exploration_rate * np.sqrt(2 * np.log(t) / count)

        return max(self.estimate(child, user=user) for child in node.children)

    def update(self, **kwargs):
        new_data = kwargs['train_df'].groupby(['user', 'missionID'])['reward'].agg(['mean', 'count'])
        self.average_rewards = new_data['mean']
        self.c = new_data['count']
        self.t = self.c.groupby('user').sum()


class MABTReeRandom(MABTreePolicy):
    def init(self, **kwargs):
        pass

    def estimate(self, _, **kwargs):
        return 0

    def update(self, **kwargs):
        pass

    def select(self, nodes, n, **kwargs):
        return set(np.random.choice(nodes, n, replace=False))
