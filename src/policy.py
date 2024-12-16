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


class PolicyWithModel(Policy, ABC):
    """
    Abstract base class for a selection policy that uses an estimator.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.round_stats = {}

    def init(self, **kwargs):
        """Reset the round stats"""
        self.round_stats = {}

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


class ModelEpsilonGreedy(PolicyWithModel):
    def __init__(self, model: nn.Module, epsilon=0.1):
        super().__init__(model)
        self.epsilon = epsilon

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


class PolicyWithMean(Policy, ABC):
    def __init__(self):
        super().__init__()
        self.df = pd.Series()

    def init(self, **kwargs):
        pass

    def estimate(self, node, **kwargs):
        user = kwargs['user']

        if node.is_leaf:
            return self.df.get((user, node.value['missionID']), 0)

        return max(self.estimate(child, user=user) for child in node.children)

    def update(self, **kwargs):
        self.df = kwargs['train_df'].groupby(
            ['user', 'missionID'])['reward'].mean()


class MeanEpsilonGreedy(PolicyWithMean):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

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


class RandomBandit(Policy):
    def init(self, **kwargs):
        pass

    def estimate(self, _, **kwargs):
        return 0

    def update(self, **kwargs):
        pass

    def select(self, nodes, n, **kwargs):
        return set(np.random.choice(nodes, n, replace=False))
