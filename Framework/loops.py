# This script contains important loops for the development pipeline

import torch


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in val_loader:
            output = model(x)
            loss = criterion(output, x)
            total_loss += loss.item()
    return total_loss / len(val_loader)
