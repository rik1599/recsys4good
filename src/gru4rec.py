import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SequentialRecommendations(Dataset):
    def __init__(self, df: pd.DataFrame, n_negative_samples=10):
        super().__init__()
        self.positives = torch.from_numpy(df['ID'].values).view(-1, 1).to(DEVICE)        
        self.histories = [torch.from_numpy(h).to(DEVICE) for h in df['history']]
        self.negatives = [torch.tensor(n).to(DEVICE) for n in df['negatives']]
        self.n_negative_samples = n_negative_samples
    
    def __len__(self):
        return len(self.positives)
    
    def __getitem__(self, idx):
        pos = self.positives[idx]
        history = self.histories[idx]

        neg = torch.randperm(len(self.negatives[idx]))[:self.n_negative_samples]
        neg = self.negatives[idx][neg]

        return history, pos, neg

def collate_with_padding(batch):
    histories, positives, negatives = zip(*batch)
    padded_histories = torch.nn.utils.rnn.pad_sequence(histories, batch_first=True, padding_value=-1, padding_side='left')
    return padded_histories, torch.stack(positives), torch.stack(negatives)


class GRU4Rec(nn.Module):
    def __init__(self, n_items, hidden_size=100, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + 1, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_items)

    def forward(self, x: torch.Tensor):
        items = x[:, :, 0] + 1 # 0 is padding
        items = self.embedding(items)

        outcomes = x[:, :, 1]
        outcomes[outcomes == -1] = 0

        x = torch.cat((items, outcomes.view(x.size(0), -1, 1)), dim=-1)

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

class BPRmax(nn.Module):
    def __init__(self, reg_lambda: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda
    
    def forward(self, positives: torch.Tensor, negatives: torch.Tensor):
        sub = positives - negatives
        negatives_weighted_softmax = F.softmax(sub, dim=-1)
        reg = torch.pow(positives, 2) * negatives_weighted_softmax * self.reg_lambda
        loss = -torch.log(F.sigmoid(sub) * negatives_weighted_softmax) + reg
        return loss.mean()

def train(model: nn.Module, dataset: SequentialRecommendations, n_epochs=10, lr=0.01):
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BPRmax()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_with_padding)

    for _ in (bar := tqdm(range(n_epochs))):
        epoch_loss = 0
        for history, pos, neg in tqdm(dataloader, leave=False):
            optimizer.zero_grad()
            out: torch.Tensor = model(history)
            pos_out = out.gather(1, pos)
            neg_out = out.gather(1, neg)
            loss = criterion(pos_out, neg_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        bar.set_postfix(loss=epoch_loss/len(dataloader))
    
    return model