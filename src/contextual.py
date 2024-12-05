import torch
import torch.nn as nn
import random
from src.policy import Policy
from torch.utils.data import DataLoader, TensorDataset

class ContextManager(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim):
        super(ContextManager, self).__init__()

        # Embedding layers with incorporated bias
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mission_embedding = nn.Embedding(num_missions, embedding_dim)

    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        dot = torch.sum(user_emb * mission_emb, dim=1)
        return dot.flatten()
    
    def fit(self, train_set, batch_size=32, epochs=10, lr=0.001, weight_decay=1e-4, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        train_set = DataLoader(train_set, batch_size)
        self.train()

        for e in range(epochs):
            for user, mission, rating in train_set:
                optimizer.zero_grad()
                output = self(user, mission)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'Epoch: {e}, Loss: {loss.item()}')

        self.eval()
        return self
    
    def get_context(self, user, mission):
        return torch.cat((self.user_embedding.weight[user], self.mission_embedding.weight[mission]))


class LinUCB(Policy):
    def __init__(self, num_users, num_arms, context_dim, alpha=1.0, device='cpu'):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            num_arms: Number of arms (actions).
            context_dim: Dimensionality of the context vectors.
            alpha: Exploration parameter.
        """
        self.num_arms = num_arms
        self.num_users = num_users
        self.context_dim = context_dim * 2
        self.alpha = alpha
        self.device = device
        
        self.context_manager = ContextManager(num_users, num_arms, context_dim).to(device)

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
            x = self.context_manager.get_context(kwargs['user'], node.value['missionID'])
            p = theta.t() @ x + self.alpha * torch.sqrt(x.t() @ A_inv @ x)
            self.round[node] = p.item()
        
        if node.is_leaf:
            return self.round[node]
        
        return max(self.estimate(child, **kwargs) for child in node.children)
    
    def update(self, **kwargs):
        train_df = kwargs['train_df']
        today = train_df[train_df['date'] == kwargs['day']]

        train_df = train_df.drop_duplicates(['user', 'missionID'], keep='last')
        train_ds = TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['missionID'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['reward'].values, dtype=torch.float, device=self.device)
        )
        self.context_manager.fit(train_ds, verbose=False)

        for _, row in today.iterrows():
            user = row['user']
            mission = row['missionID']
            reward = row['reward']
            x = self.context_manager.get_context(user, mission)
            self.A[mission] += x @ x.t()
            self.b[mission] += reward * x