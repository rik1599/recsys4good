import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --------- TRAINING UTILITIES --------- #
def _train(
        model: nn.Module, 
        train_set: Dataset, 
        batch_size=32, 
        epochs=10, 
        lr=0.01, 
        weight_decay=0.0,
        verbose=True):
    """
    Train a PyTorch model on a dataset.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_set = DataLoader(train_set, batch_size)
    model.train()

    for _ in (t := tqdm(range(epochs))) if verbose else range(epochs):
        for user, mission, rating in train_set:
            optimizer.zero_grad()
            output = model(user, mission)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            if verbose:
                t.set_postfix(loss=loss.item())

    return model


# --------- MODELS --------- #
class MF(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim):
        super(MF, self).__init__()

        # Embedding layers with incorporated bias
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
    
    def fit(self, train_set, batch_size=32, epochs=10, lr=0.01, weight_decay=0.0, verbose=True):
        return _train(self, train_set, batch_size, epochs, lr, weight_decay, verbose)