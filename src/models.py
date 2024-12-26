import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

class MF(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim, **kwargs):
        super(MF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mission_embedding = nn.Embedding(num_missions, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.mission_bias = nn.Embedding(num_missions, 1)

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.mission_embedding.weight)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.mission_bias.weight)

        # Training parameters
        self.device = kwargs.get('device', 'cpu')
        self.lr = kwargs.get('lr', 1e-2)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 30)
        self.batch_size = kwargs.get('batch_size', 32)

    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        user_bias = self.user_bias(user)
        mission_bias = self.mission_bias(mission)
        
        dot = torch.sum(user_emb * mission_emb, dim=1, keepdim=True) + user_bias + mission_bias
        dot = dot.view(-1)

        if self.training:
            return dot
        return torch.relu(dot)
    
    def fit(self, train_df: pd.DataFrame):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_df = train_df.drop_duplicates(subset=['user', 'missionID'], keep='last')
        train_dl = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['missionID'].values, dtype=torch.long, device=self.device),
            torch.tensor(train_df['reward'].values, dtype=torch.float, device=self.device)
        ), self.batch_size, shuffle=True)

        self.to(self.device)
        self.train()
        for _ in (t := tqdm(range(self.epochs), leave=False)):
            epoch_loss = 0
            for user, mission, rating in train_dl:
                optimizer.zero_grad()
                output = self(user, mission)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            t.set_postfix({'loss': epoch_loss / len(train_dl)})
        
        self.eval()
        return self

    def predict(self, user, mission):
        user = torch.tensor(user, dtype=torch.long, device=self.device)
        mission = torch.tensor(mission, dtype=torch.long, device=self.device)
        return self(user, mission)
    
