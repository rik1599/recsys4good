import torch
from torch.utils.data import Dataset


class MissionDataset(Dataset):
    def __init__(self, missions, users, ratings):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.missions = torch.from_numpy(missions).to(device).long()
        self.users = torch.from_numpy(users).to(device).long()
        self.ratings = torch.from_numpy(ratings).to(device).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.missions[idx], self.ratings[idx]
