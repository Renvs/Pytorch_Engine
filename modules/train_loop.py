import torch
import tqdm

from torch import nn, optim
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model: nn.Module,
               train_data: torch.utils.data.DataLoader,
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               accuracy,
               device: str = device 
):
    model.to(device)
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (x, y) in tqdm(enumerate(train_data)):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        accuracy.update(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_data)
    train_acc = accuracy.compute()
    accuracy.reset()

    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.3f}%")

    return train_loss, train_acc

def test_loop( model: nn.Module,
               test_data: torch.utils.data.DataLoader,
               loss_fn: nn.Module, 
               accuracy,
               save_path,
               device: str = device 
):
    model.to(device)
    model.train()

    test_loss, test_acc = 0, 0

    for x, y in tqdm(test_data):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        accuracy.update(pred, y)
    
    test_loss /= len(test_data)
    test_acc = accuracy.compute()
    accuracy.reset()
    print(f"Train loss: {test_loss:.4f} | Train accuracy: {test_acc * 100:.3f}%")

    return test_loss, test_acc
