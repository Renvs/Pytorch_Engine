import torch
import tqdm
import create_summary as create_summary
import copy
import modules.data_loader as data_loader

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Tuple, List, Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: nn.Module,
               train_data: torch.utils.data.Dataset,
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               scheduler: optim.lr_scheduler,
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
        scheduler.step()
    
    train_loss = train_loss / len(train_data)
    train_acc = accuracy.compute()
    accuracy.reset()

    return train_loss, train_acc

def test_step( model: nn.Module,
               test_data: torch.utils.data.Dataset,
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
           model_name: str,
           train_data: torch.utils.data.Dataset, 
           test_data: torch.utils.data.Dataset,
           loss_fn: nn.Module, 
           optimizer: optim.Optimizer, 
           scheduler: optim.lr_scheduler,
           accuracy,
           model_path: str,
           epochs: int,
           patience: int,
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
    patience_counter = 0
    best_weights = None

    for epoch in tqdm(range(epochs)):

        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_step(
            model=model, 
            train_data=train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
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


        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc.item())
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc.item())

        current_min_loss = min(result['test_loss'])
        if current_min_loss < best_loss:
            best_loss = current_min_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            print(f'Save best weights with loss: {current_min_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        data_loader.save_models(model, model_path, model_name)

    print(f"best_train_loss = {min(result['train_loss'])}")
    print(f"best_train_acc = {max(result['train_acc'])}")
    print(f"best_test_loss = {min(result['test_loss'])}")
    print(f"best_test_acc = {max(result['test_acc'])}")

    return result

def summary_writer_addon(
        model: nn.Module,
        model_name: str,
        model_path: str,
        train_data: torch.utils.data.Dataset, 
        test_data: torch.utils.data.Dataset,
        loss_fn: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler,
        accuracy,
        epochs: int,
        batch_size: int, 
        image_size: int,
        writer: create_summary,
        device: str = device,
        patience: int = 10
) -> Dict[str, List]:

    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        }
    
    model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in tqdm(range(epochs)):

        print(f"Epoch {epoch + 1}/{epochs}\n")
        
        train_loss, train_acc = train_step(
            model=model, 
            train_data=train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
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
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.3f}%\n")

        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc.item())
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc.item())

        current_min_loss = min(result['test_loss'])

        if current_min_loss < best_loss:
            best_loss = current_min_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            print(f'Save best weights with loss: {current_min_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break


        if writer:
            
            writer.add_scalars(
                main_tag= 'Loss',
                tag_scalar_dict={'train_loss': train_loss, 'test_loss': test_loss}, 
                global_step= epoch
            )
            
            writer.add_scalars(
                main_tag= 'Accuracy',
                tag_scalar_dict= {'train_accuracy': train_acc, 'test_accuracy': test_acc}, 
                global_step= epoch
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)
        data_loader.save_models(model, model_path, model_name)
        
    writer.add_graph(model=model, 
                     input_to_model= torch.randn(batch_size, 3, image_size, image_size).to(device))
    writer.close()

    print(f"\nbest_train_loss = {min(result['train_loss'])}")
    print(f"best_train_acc = {max(result['train_acc'])}")
    print(f"best_test_loss = {min(result['test_loss'])}")
    print(f"best_test_acc = {max(result['test_acc'])}\n")

    return result