import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.dataset import MissionDataset, DEVICE
from .policy import Policy
from src.tree import TreeNode
from src.utils import train

class EpsilonGreedyWithModel(Policy):
    def __init__(self, model: nn.Module, epsilon=0.1):
        super().__init__()
        np.random.seed(42)
        self.model = model
        self.epsilon = epsilon
        self.__round_stats = {}
    
    def init(self, **kwargs):
        """Reset the round stats"""
        self.__round_stats = {}

    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        selectable_nodes = {node: self.__retrieve_node_data(user, node) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            if np.random.rand() <= self.epsilon:
                node = np.random.choice(list(selectable_nodes.keys()))
            else:
                node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(node)
            selectable_nodes.pop(node)
        
        return selected_nodes
    
    def __retrieve_node_data(self, user, node: TreeNode):
        # calculate the expected reward of the node for this round if not already calculated
        if node not in self.__round_stats and node.is_leaf:
            ids = torch.tensor([node.value['missionID']], dtype=torch.long, device=DEVICE)
            user = torch.tensor([user], dtype=torch.long, device=DEVICE)
            data = self.model(user, ids).item()
            self.__round_stats[node] = data
        
        if node.is_leaf:
            return self.__round_stats[node]
        
        return max(self.__retrieve_node_data(user, child) for child in node.children)
    
    def update(self, **kwargs):
        train_df: pd.DataFrame = kwargs['train_df']
        dataset = MissionDataset(train_df['missionID'].values, train_df['user'].values, train_df['reward'].values)

        epochs = kwargs.get('epochs', 15)
        lr = kwargs.get('lr', 0.001)
        batch_size = kwargs.get('batch_size', 16)
        self.model = train(self.model, dataset, epochs=epochs, lr=lr, batch_size=batch_size, verbose=False)