import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MissionDataset, DEVICE
from tqdm.auto import tqdm

def train(
        model: nn.Module, 
        train_set: MissionDataset, 
        validation_set: MissionDataset = None, 
        batch_size=32, 
        epochs=10, 
        lr=0.01, 
        weight_decay=0.0):
    """
    Train a PyTorch model on a dataset.
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # balance positive and negative samples
    pos_weight = (train_set.ratings == 0).sum() / (train_set.ratings == 1).sum()
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