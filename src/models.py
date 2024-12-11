import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

class MF(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim):
        super(MF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mission_embedding = nn.Embedding(num_missions, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.mission_bias = nn.Embedding(num_missions, 1)

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.mission_embedding.weight)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.mission_bias.weight)

    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        user_bias = self.user_bias(user)
        mission_bias = self.mission_bias(mission)
        
        dot = torch.sum(user_emb * mission_emb, dim=1, keepdim=True) + user_bias + mission_bias
        return dot.squeeze()
    
    def fit(self, train_df: pd.DataFrame, **kwargs):
        lr = kwargs.get('lr', 0.01)
        weight_decay = kwargs.get('weight_decay', 0.0)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        device = kwargs.get('device', 'cpu')

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        train_dl = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long, device=device),
            torch.tensor(train_df['missionID'].values, dtype=torch.long, device=device),
            torch.tensor(train_df['reward'].values, dtype=torch.float, device=device)
        ), batch_size, shuffle=True)

        self.to(device)
        self.train()
        for _ in (t := tqdm(range(epochs), leave=False)):
            epoch_loss = 0
            for user, mission, rating in train_dl:
                optimizer.zero_grad()
                output = self(user, mission)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            t.set_postfix({'loss': epoch_loss / len(train_dl)})
        
        print(f'Final loss: {epoch_loss / len(train_dl)}')
        self.eval()
        return self


class MLP(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim, hidden_dim, dropout):
        super(MLP, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mission_embedding = nn.Embedding(num_missions, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.mission_embedding.weight)
    
    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        
        mlp_input = torch.cat((user_emb, mission_emb), dim=1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out.squeeze()
    
    def fit(self, train_df: pd.DataFrame, **kwargs):
        lr = kwargs.get('lr', 0.01)
        weight_decay = kwargs.get('weight_decay', 0.0)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        device = kwargs.get('device', 'cpu')

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        train_dl = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long, device=device),
            torch.tensor(train_df['missionID'].values, dtype=torch.long, device=device),
            torch.tensor(train_df['reward'].values, dtype=torch.float, device=device)
        ), batch_size, shuffle=True)

        self.to(device)
        self.train()
        for _ in (t := tqdm(range(epochs), leave=False)):
            epoch_loss = 0
            for user, mission, rating in train_dl:
                optimizer.zero_grad()
                output = self(user, mission)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            t.set_postfix({'loss': epoch_loss / len(train_dl)})
        
        print(f'Final loss: {epoch_loss / len(train_dl)}')
        self.eval()
        return self


class AutoRec(nn.Module):
    def __init__(self, d, k, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, k),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, d)
        )
    
    def forward(self, r):
        return self.decoder(self.encoder(r))

    def fit(self, train_df: pd.DataFrame, **kwargs):
        lr = kwargs.get('lr', 0.01)
        weight_decay = kwargs.get('weight_decay', 0.0)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        device = kwargs.get('device', 'cpu')
        n_users = train_df['user'].max() + 1

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        user_item_matrix, na_mask = self.__get_dataset(train_df, n_users)
        self.user_item_matrix = torch.tensor(user_item_matrix, dtype=torch.float, device=device)

        train_dl = DataLoader(TensorDataset(
            self.user_item_matrix,
            torch.tensor(na_mask, dtype=torch.float, device=device)
        ), batch_size, shuffle=True)

        self.to(device)
        self.train()
        for _ in (t := tqdm(range(epochs), leave=False)):
            epoch_loss = 0
            for r, mask in train_dl:
                optimizer.zero_grad()
                output = self(r)
                loss = criterion(r, output * mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            t.set_postfix({'loss': epoch_loss / len(train_dl)})

        print(f'Final loss: {epoch_loss / len(train_dl)}')
        self.eval()
        return self
    
    def __get_dataset(self, df: pd.DataFrame, n_users: int):
        user_item_matrix = df.pivot(index='user', columns='missionID', values='reward')

        x = pd.DataFrame(
            index=range(n_users),
            columns=user_item_matrix.columns,
            dtype=float
        )
        x.update(user_item_matrix)

        na_mask = ~x.isna().values
        x.fillna(0, inplace=True)

        return x.values, na_mask
    
    def predict(self, user: torch.Tensor, mission: torch.Tensor):
        x = self.user_item_matrix[user]
        ratings = self(x)
        return ratings[range(len(mission)), mission]

    
