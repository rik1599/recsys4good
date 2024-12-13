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
        self.lr = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 20)
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


class MLP(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim, hidden_dim, dropout, **kwargs):
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

        # Training parameters
        self.device = kwargs.get('device', 'cpu')
        self.lr = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 20)
        self.batch_size = kwargs.get('batch_size', 32)
    
    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        
        mlp_input = torch.cat((user_emb, mission_emb), dim=1)
        mlp_out = self.mlp(mlp_input)
        mlp_out = mlp_out.view(-1)

        if self.training:
            return mlp_out
        return torch.relu(mlp_out)
    
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


class UserBasedAutoRec(nn.Module):
    def __init__(self, n_users, n_missions, hidden_dim, dropout, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_missions, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_missions)
        )

        # Training parameters
        self.device = kwargs.get('device', 'cpu')
        self.lr = kwargs.get('lr', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)

        self.user_item_matrix = torch.zeros(n_users, n_missions, device=self.device)
        
    def forward(self, r):
        return self.decoder(self.encoder(r))

    def fit(self, train_df: pd.DataFrame):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        user_item_matrix, na_mask = self.__update_user_item_matrix(train_df)
        self.user_item_matrix = torch.tensor(user_item_matrix, dtype=torch.float, device=self.device)

        train_dl = DataLoader(TensorDataset(
            self.user_item_matrix,
            torch.tensor(na_mask, dtype=torch.float, device=self.device)
        ), self.batch_size, shuffle=True)

        self.to(self.device)
        self.train()
        for _ in (t := tqdm(range(self.epochs), leave=False)):
            epoch_loss = 0
            for r, mask in train_dl:
                optimizer.zero_grad()
                output = self(r)
                loss = criterion(r, output * mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            t.set_postfix({'loss': epoch_loss / len(train_dl)})

        self.eval()
        return self
    
    def __update_user_item_matrix(self, df: pd.DataFrame):
        user_item_matrix = df \
            .drop_duplicates(subset=['user', 'missionID'], keep='last') \
            .pivot(index='user', columns='missionID', values='reward')

        x = pd.DataFrame(
            index=range(self.user_item_matrix.shape[0]),
            columns=range(self.user_item_matrix.shape[1]),
            dtype=float,
        )
        x.update(user_item_matrix)

        na_mask = ~x.isna().values
        x.fillna(0, inplace=True)
        
        return x.values, na_mask
    
    def predict(self, user, mission):
        x = self.user_item_matrix[user]
        ratings = self(x)
        return ratings[range(len(mission)), mission]

    
