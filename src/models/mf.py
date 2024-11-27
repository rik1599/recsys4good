import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MissionDataset, DEVICE
from tqdm.auto import tqdm


class MissionMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_missions, embedding_dim):
        torch.manual_seed(42)
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
