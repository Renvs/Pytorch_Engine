import torch
import tqdm

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Tuple, List, Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: nn.Module,
               train_data: torch.utils.data.DataLoader,
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               accuracy,
               device: str = device 
) -> Tuple[float, float]:
    
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
    
    train_loss = train_loss / len(train_data)
    train_acc = accuracy.compute()
    accuracy.reset()

    return train_loss, train_acc

def test_step( model: nn.Module,
               test_data: torch.utils.data.DataLoader,
               loss_fn: nn.Module, 
               accuracy,
               device: str = device 
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch,(x, y) in tqdm(enumerate(test_data)):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            accuracy.update(pred, y)
    
        test_loss = test_loss / len(test_data)
        test_acc = accuracy.compute()
        accuracy.reset()

    return test_loss, test_acc

def train( model: nn.Module,
           train_data: torch.utils.data.DataLoader, 
           test_data: torch.utils.data.DataLoader,
           loss_fn: nn.Module, 
           optimizer: optim.Optimizer, 
           accuracy,
           model_path: str,
           epochs: int,
           device: str = device
) -> Dict[str, List]:
    
    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        }
    
    model.to(device)

    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_step(
            model=model, 
            train_data=train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy=accuracy,
            device=device
        )
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.3f}%")

        test_loss, test_acc = test_step(
            model=model,
            test_data=test_data,
            loss_fn=loss_fn,
            accuracy=accuracy,
            device=device
        )
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.3f}%")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f'Save at {model_path} Loss: {test_loss:.4f}')

        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc.item())
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc.item())

    print(f"best_train_loss = {min(result['train_loss'])}")
    print(f"best_train_acc = {max(result['train_acc'])}")
    print(f"best_test_loss = {min(result['test_loss'])}")
    print(f"best_test_acc = {max(result['test_acc'])}")

    return result

def single_tracking(
        model: nn.Module,
        train_data: torch.utils.data.DataLoader, 
        test_data: torch.utils.data.DataLoader,
        loss_fn: nn.Module, 
        optimizer: optim.Optimizer, 
        accuracy,
        model_path: str,
        epochs: int,
        batch_size: int, 
        image_size: int,
        device: str = device
) -> Dict[str, List]:
    
    writer = SummaryWriter()

    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        }
    
    model.to(device)

    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_step(
            model=model, 
            train_data=train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy=accuracy,
            device=device
        )
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.3f}%")

        test_loss, test_acc = test_step(
            model=model,
            test_data=test_data,
            loss_fn=loss_fn,
            accuracy=accuracy,
            device=device
        )
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.3f}%")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f'Save at {model_path} Loss: {test_loss:.4f}')

        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc.item())
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc.item())

        writer.add_scalars(main_tag= 'Loss',
                        tag_scalar_dict={'train_loss': train_loss, 'test_loss': test_loss}, 
                        global_step= epoch
                        )
        
        writer.add_scalars(main_tag= 'Accuracy',
                        tag_scalar_dict= {'train_accuracy': train_loss, 'test_accuracy': test_loss}, 
                        global_step= epoch
                        )
        
    writer.add_graph(model=model, 
                     input_to_model= torch.randn(batch_size, 3, image_size, image_size).to(device))
    writer.close()

    print(f"best_train_loss = {min(result['train_loss'])}")
    print(f"best_train_acc = {max(result['train_acc'])}")
    print(f"best_test_loss = {min(result['test_loss'])}")
    print(f"best_test_acc = {max(result['test_acc'])}")

    return result