import torch
import pandas as pd

class ContextManager:
    def __init__(self, n_users, n_features, device='cpu'):
        self.contexts = pd.DataFrame(
            index=range(n_users),
            columns=range(n_features),
            data=0,
            dtype=float,
        )
        self.device = device
        self.context_dim = n_features
    
    def update(self, df: pd.DataFrame):

        user_item_matrix = df \
            .drop_duplicates(['user', 'missionID'], keep='last') \
            .pivot(index='user', columns='missionID', values='reward') \
            .fillna(0)
        self.contexts.update(user_item_matrix)
    
    def get(self, user):
        return torch.tensor(self.contexts.loc[user].values, dtype=torch.float32, device=self.device)
    

class LinUCB:
    def __init__(self, num_arms, context_manager: ContextManager, alpha=1.0, device='cpu'):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            num_arms: Number of arms (actions).
            context_dim: Dimensionality of the context vectors.
            alpha: Exploration parameter.
        """
        self.num_arms = num_arms
        self.context_dim = context_manager.context_dim
        self.alpha = alpha
        self.device = device
        self.context_manager = context_manager
        
        # Initialize arm-specific parameters
        self.A = [torch.eye(self.context_dim, device=self.device) for _ in range(num_arms)]  # (num_arms, context_dim, context_dim)
        self.b = [torch.zeros(self.context_dim, device=self.device) for _ in range(num_arms)]
    

    def select(self, user):
        x = self.context_manager.get(user)

        p = torch.zeros(self.num_arms, device=self.device)
        for a in range(self.num_arms):
            A_inv = torch.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta.t() @ x + self.alpha * torch.sqrt(x.t() @ A_inv @ x)
        
        return p.argsort(descending=True).tolist()


    def update(self, train_df: pd.DataFrame, day: int):
        today = train_df[train_df['createdAt'] == day]
        for _, row in today.iterrows():
            a = row['missionID']

            x = self.context_manager.get(row['user'])

            self.A[a] += torch.outer(x, x)
            self.b[a] += row['reward'] * x

        self.context_manager.update(train_df)
