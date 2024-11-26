import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MissionDataset(Dataset):
    def __init__(self, missions, users, ratings):
        self.missions = torch.from_numpy(missions).to(DEVICE).long()
        self.users = torch.from_numpy(users).to(DEVICE).long()
        self.ratings = torch.from_numpy(ratings).to(DEVICE).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.missions[idx], self.ratings[idx]

class MissionMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim):
        super(MissionMatrixFactorization, self).__init__()
        
        # Embedding layers with incorporated bias
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mission_embedding = nn.Embedding(num_missions, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.mission_bias = nn.Embedding(num_missions, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, mission):
        user_emb = self.user_embedding(user)
        mission_emb = self.mission_embedding(mission)
        user_bias = self.user_bias(user)
        mission_bias = self.mission_bias(mission)
        dot = torch.sum(user_emb * mission_emb, dim=1) + user_bias.squeeze() + mission_bias.squeeze() + self.bias
        dot = dot.flatten()
        return dot

def fit(model: nn.Module, train_set: MissionDataset, batch_size=32, epochs=10, lr=0.01, weight_decay=0.0, validation_set: Dataset=None):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = (train_set.ratings == 0).sum() / (train_set.ratings == 1).sum() # balance positive and negative samples
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(-1))
    train_set = DataLoader(train_set, batch_size)
    if validation_set:
        validation_set = DataLoader(validation_set, batch_size)
    
    for _ in (bar := tqdm(range(epochs))):
        model.train()
        train_loss = 0
        for user, mission, rating in tqdm(train_set, leave=False):
            optimizer.zero_grad()
            prediction = model(user, mission)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if not validation_set:
            bar.set_postfix(loss=train_loss / len(train_set))

        if validation_set:
            model.eval()
            with torch.no_grad():
                validation_loss = 0
                for user, mission, rating in validation_set:
                    prediction = model(user, mission)
                    loss = criterion(prediction, rating)
                    validation_loss += loss.item()
            bar.set_postfix(loss=train_loss / len(train_set), val_loss=validation_loss / len(validation_set))
    return model