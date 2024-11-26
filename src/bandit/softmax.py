from policy import Policy
from src.tree import TreeNode
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.utils import train
from src.dataset import MissionDataset, DEVICE
from scipy.special import softmax

class SoftmaxBandit(Policy):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.__round_stats = {}

    def init(self, **kwargs):
        """Reset the round stats"""
        self.__round_stats = {}

    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        p = softmax([self.__retrieve_node_data(user, node) for node in nodes])
        selected_nodes = np.random.choice(nodes, n, p=p, replace=False)
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
        dataset = MissionDataset(train_df['missionID'].values, train_df['user'].values, train_df['performance'].values)
        self.model = train(self.model, dataset)