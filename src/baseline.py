import torch
import numpy as np
import pandas as pd
from .policy import Policy


class ContextManager:
    def __init__(self, n_users, n_features, device='cpu'):
        self.n_users = n_users
        self.n_features = n_features
        self.device = device
        self.context_dim = n_features
    
    def init(self):
        self.contexts = pd.DataFrame(
            index=range(self.n_users),
            columns=range(self.n_features),
            data=0,
            dtype=float,
        )

    def update(self, df: pd.DataFrame):

        user_item_matrix = df \
            .drop_duplicates(['user', 'missionID'], keep='last') \
            .pivot(index='user', columns='missionID', values='reward') \
            .fillna(0)
        self.contexts.update(user_item_matrix)

    def get(self, user):
        return torch.tensor(self.contexts.loc[user].values, dtype=torch.float32, device=self.device)


class LinUCB(Policy):
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

    def init(self, **kwargs):
        self.A = [torch.eye(self.context_dim, device=self.device) for _ in range(self.num_arms)]
        self.b = [torch.zeros(self.context_dim, device=self.device) for _ in range(self.num_arms)]
        self.context_manager.init()

    def select(self, user):
        x = self.context_manager.get(user)

        p = torch.zeros(self.num_arms, device=self.device)
        for a in range(self.num_arms):
            A_inv = torch.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta.t() @ x + self.alpha * torch.sqrt(x.t() @ A_inv @ x)

        return p.argsort(descending=True).tolist()

    def estimate(self, node, **kwargs):
        pass

    def update(self, train_df: pd.DataFrame, day: int):
        today = train_df[train_df['createdAt'] == day]
        for _, row in today.iterrows():
            a = row['missionID']

            x = self.context_manager.get(row['user'])

            self.A[a] += torch.outer(x, x)
            self.b[a] += row['reward'] * x

        self.context_manager.update(train_df)


class EpsilonGreedy(Policy):
    def __init__(self, num_arms, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.num_arms = num_arms

    def init(self, **kwargs):
        self.average_rewards = pd.Series()

    def select(self, user):
        selectable = {
            arm: self.average_rewards.get((user, arm), 0)
            for arm in range(self.num_arms)
        }
        rank = []

        for _ in range(self.num_arms):
            if np.random.rand() < self.epsilon:
                x = np.random.choice(list(selectable.keys()))
            else:
                x = max(selectable, key=selectable.get)
            rank.append(x)
            selectable.pop(x)

        return rank

    def estimate(self, node, **kwargs):
        pass

    def update(self, **kwargs):
        self.average_rewards = kwargs['train_df'].groupby(
            ['user', 'missionID'])['reward'].mean()


class UCB1(Policy):
    def __init__(self, num_arms, exploration_rate=1):
        super().__init__()
        self.exploration_rate = exploration_rate
        self.num_arms = num_arms

    def init(self, **kwargs):
        self.average_rewards = pd.Series()
        self.c = pd.Series()
        self.t = pd.Series()

    def select(self, user):
        arms = np.array([self.estimate(arm, user)
                        for arm in range(self.num_arms)])
        return arms.argsort()[::-1].tolist()

    def estimate(self, arm, user):
        mean = self.average_rewards.get((user, arm), 0)
        count = self.c.get((user, arm), 0)  # avoid division by zero
        t = self.t.get(user, 0)  # avoid log(0)

        if count == 0 or t == 0:
            return np.inf

        return mean + self.exploration_rate * np.sqrt(np.log(t) / count)

    def update(self, **kwargs):
        new_data = kwargs['train_df'].groupby(['user', 'missionID'])[
            'reward'].agg(['mean', 'count'])
        self.average_rewards = new_data['mean']
        self.c = new_data['count']
        self.t = self.c.groupby('user').sum()
