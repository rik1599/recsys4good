import torch
import pandas as pd
from src.policy import Policy

class ContextManager:
    def __init__(self, n_users, features, device='cpu'):
        self.contexts = pd.DataFrame(
            index=range(n_users),
            columns=features,
            data=0,
            dtype=float,
        )
        self.device = device
    
    def update(self, df: pd.DataFrame):
        user_item_matrix = df.pivot(index='user', columns='missionID', values='reward').fillna(0)
        self.contexts.update(user_item_matrix)
    
    def get(self, user):
        return torch.tensor(self.contexts.loc[user].values, dtype=torch.float32, device=self.device)
    

class LinUCB(Policy):
    def __init__(self, num_users, num_arms, context_dim, context_manager: ContextManager, alpha=1.0, device='cpu'):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            num_arms: Number of arms (actions).
            context_dim: Dimensionality of the context vectors.
            alpha: Exploration parameter.
        """
        self.num_arms = num_arms
        self.num_users = num_users
        self.context_dim = context_dim
        self.alpha = alpha
        self.device = device
        self.context_manager = context_manager
        
        # Initialize arm-specific parameters
        self.A = torch.eye(self.context_dim, device=self.device).repeat(num_arms, 1, 1)  # (num_arms, context_dim, context_dim)
        self.b = torch.zeros(num_arms, self.context_dim, device=self.device)  # (num_arms, context_dim)

    def init(self, **kwargs):
        self.round = {}
    
    def select(self, nodes, n, **kwargs):
        user = kwargs['user']
        selectable_nodes = {node: self.estimate(node, user=user) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            selected_node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(selected_node)
            selectable_nodes.pop(selected_node)
        
        return selected_nodes
    
    def estimate(self, node, **kwargs):
        if node not in self.round and node.is_leaf:
            A_inv = torch.inverse(self.A[node.value['missionID']])
            theta = A_inv @ self.b[node.value['missionID']]
            x = self.context_manager.get(kwargs['user'])
            p = theta.t() @ x + self.alpha * torch.sqrt(x.t() @ A_inv @ x)
            self.round[node] = p.item()
        
        if node.is_leaf:
            return self.round[node]
        
        return max(self.estimate(child, **kwargs) for child in node.children)
    
    def update(self, **kwargs):
        train_df: pd.DataFrame = kwargs['train_df']
        today: pd.DataFrame = train_df[train_df['date'] == kwargs['day']]

        for _, row in today.iterrows():
            user = row['user']
            mission = row['missionID']
            reward = row['reward']
            x = self.context_manager.get(user)
            self.A[mission] += x @ x.t()
            self.b[mission] += reward * x
        
        train_df = train_df.drop_duplicates(['user', 'missionID'], keep='last')
        self.context_manager.update(train_df)