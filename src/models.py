import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MissionDataset
from tqdm.auto import tqdm

# --------- TRAINING UTILITIES --------- #
def train(
        model: nn.Module, 
        train_set: MissionDataset, 
        validation_set: MissionDataset = None, 
        batch_size=32, 
        epochs=10, 
        lr=0.01, 
        weight_decay=0.0,
        verbose=True):
    """
    Train a PyTorch model on a dataset.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # balance positive and negative samples
    pos_weight = (train_set.ratings == 0).sum() / (train_set.ratings == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(-1))

    train_set = DataLoader(train_set, batch_size)
    if validation_set:
        validation_set = DataLoader(validation_set, batch_size)

    bar = tqdm(range(epochs)) if verbose else range(epochs)
    for _ in bar:
        model.train()
        train_loss = 0

        epoch_bar = tqdm(train_set, leave=False) if verbose else train_set
        for user, mission, rating in epoch_bar:
            optimizer.zero_grad()
            prediction = model(user, mission)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if not validation_set and verbose:
            bar.set_postfix(loss=train_loss / len(train_set))

        if validation_set:
            model.eval()
            with torch.no_grad():
                validation_loss = 0
                for user, mission, rating in validation_set:
                    prediction = model(user, mission)
                    loss = criterion(prediction, rating)
                    validation_loss += loss.item()
            if verbose:
                bar.set_postfix(loss=train_loss / len(train_set), val_loss=validation_loss / len(validation_set))
    return model


# --------- MODELS --------- #
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
        dot = torch.sum(user_emb * mission_emb, dim=1) + \
            user_bias.squeeze() + mission_bias.squeeze() + self.bias
        return dot.flatten()


class MissionLinearRegression(nn.Module):
    def __init__(self, num_users, num_missions):
        super(MissionLinearRegression, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 1)
        self.mission_embedding = nn.Embedding(num_missions, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, mission):
        user_emb = self.user_embedding(user).squeeze()
        mission_emb = self.mission_embedding(mission).squeeze()
        dot = user_emb + mission_emb + self.bias
        return dot.flatten()