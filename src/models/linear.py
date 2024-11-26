import torch
from torch import nn

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