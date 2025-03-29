import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'\nEpoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss = []
    batch_train_acc = []

    for batch in train_dataloader:
        if len(batch) != 2:
            raise ValueError("Expected batch to contain two elements (inputs, targets), but got {len(batch)}.")

        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_train_loss.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        _, true_labels = torch.max(targets, 1)  
        total = targets.size(0)
        correct = (predicted == true_labels).sum().item()
        batch_train_acc.append(correct / total)

    return batch_train_loss, batch_train_acc


def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []

    with torch.no_grad(): 
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
            accuracy = correct / len(target)
            batch_test_acc.append(accuracy)

    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")
