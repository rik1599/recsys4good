from src.policy import Policy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class MF(nn.Module):
    def __init__(self, n_users, n_arms, context_dim, device='cpu'):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, context_dim)
        self.arm_embeddings = nn.Embedding(n_arms, context_dim)
        self.device = device
        self.to(device)
    
    def forward(self, user, arm):
        return torch.sum(self.user_embeddings(user) * self.arm_embeddings(arm), dim=1)
    
    def fit(self, train_df: pd.DataFrame, epochs=10, lr=1e-3, weight_decay=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        dataloader = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['missionID'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['performance'].values, dtype=torch.float, device=self.device)
        ), batch_size=16, shuffle=True)
        
        for _ in range(epochs):
            self.train()
            for user, arm, reward in dataloader:
                optimizer.zero_grad()
                prediction = self(user, arm)
                loss = criterion(prediction, reward)
                loss.backward()
                optimizer.step()
        
        self.eval()


class LinUCB(Policy):
    def __init__(self, n_users, n_arms, context_dim, alpha, device='cpu'):
        self.device = device
        self.model = MF(n_users, n_arms, context_dim, device=device)
        self.context_dim = context_dim
        self.alpha = alpha

        self.A = torch.eye(context_dim, device=device).repeat(n_arms, 1, 1)
        self.b = torch.zeros(n_arms, context_dim, device=device)
    
    def init(self, **kwargs):
        self._ucb = {}
    
    def select(self, nodes, n, **kwargs):
        # Get features for the user
        x = torch.tensor(kwargs['user'], dtype=torch.int, device=self.device)
        x = self.model.user_embeddings(x)

        selectable_nodes = {node: self.estimate(node, context=x) for node in nodes}
        selected_nodes = set()

        for _ in range(n):
            node = max(selectable_nodes, key=selectable_nodes.get)
            selected_nodes.add(node)
            selectable_nodes.pop(node)
        
        return selected_nodes

    def estimate(self, node, **kwargs):
        x: torch.Tensor = kwargs['context']
        if node not in self._ucb and node.is_leaf:
            A_inv = torch.inverse(self.A[node.value['missionID']])
            theta = A_inv @ self.b[node.value['missionID']]
            ucb = torch.dot(theta, x) + self.alpha * torch.sqrt(x @ A_inv @ x)
            self._ucb[node] = ucb.item()
        
        if node.is_leaf:
            return self._ucb[node]
        
        return max(self.estimate(child, context=x) for child in node.children)
    
    def update(self, **kwargs):
        # Fit embeddings model
        train_df: pd.DataFrame = kwargs['train_df']
        train_df = train_df.drop_duplicates(subset=['user', 'missionID'], keep='last')
        self.model.fit(train_df)

        # Update LinUCB model
        actions_df = kwargs['actions_df']
        for _, row in actions_df.iterrows():
            x = torch.tensor(row['user'], dtype=torch.int, device=self.device)
            x = self.model.user_embeddings(x)
            self.A[row['missionID']] += torch.outer(x, x)
            self.b[row['missionID']] += row['reward'] * x