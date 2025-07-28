import torch 
import os
import torchvision.models as models

from torchinfo import summary
from torch import nn, optim
from modules import data_setup, get_data, train_loop

NUM_WORKERS = os.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def feature_extraction(model_1: nn.Module,
                       file_name: str, 
                       data_path: str, 
                       save_path: str,
                       url: str, 
                       weight: models,
                       accuracy,
                       loss_fn: nn.Module,
                       batch_size: int,
                       optimizer_class: optim.Optimizer = optim.AdamW,
                       device: str = device,
                       num_workers: int = NUM_WORKERS,
                       epochs: int = 10,
                       ):
    train_dir, test_dir, model_path = get_data.get_data(
        url= url, data_path= data_path, save_path= save_path, filename= file_name
    )

    weights = weight

    train_dataloader, test_dataloader, class_names = data_setup.load_dataloader(
        train_dir, test_dir, weights, batch_size, num_workers
    )

    crop_size = weights.transforms().crop_size[0]
    model = model_1.to(device)
    original_classifier = model.classifier
    dropout_layer = original_classifier[0]
    dropout_p = dropout_layer.p
    dropout_inplace = dropout_layer.inplace

    print(model.classifier)

    for params in model.features.parameters():
        params.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=dropout_inplace),
        nn.Linear(in_features= model.classifier[1].in_features, out_features= len(class_names)),
    )

    dummy_test = summary(
        model, 
        input_size= (1, 3, crop_size, crop_size),
        col_names= ['input_size', 'output_size', 'num_params', 'trainable'],
        col_width= 20,
        row_settings=['var_names']
    )

    dummy_test

    loss_fn = loss_fn
    train_accuracy = accuracy.to(device)
    test_accuracy = accuracy.to(device)
    optimizer = optimizer_class

    best_loss = float('inf')

    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    for epoch in range(epochs):
        print(f'\n[INFO] Epoch: {epoch + 1}/{epochs}\n')
        
        train_loss, train_acc = train_loop.train_loop(
            model, train_dataloader, loss_fn, optimizer, train_accuracy, device
        )

        test_loss, test_acc = train_loop.test_loop(
            model, test_dataloader, loss_fn, test_accuracy, save_path, device
        )

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f'Save at {model_path} Loss: {test_loss:.4f}')

        current_result = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }

        for key, value in current_result.items():
            result[key].append(value)

    print(f"best_train_loss = {min(result['train_loss'])}")
    print(f"best_train_acc = {max(result['train_acc'])}")
    print(f"best_test_loss = {min(result['test_loss'])}")
    print(f"best_test_acc = {max(result['test_acc'])}")

    return result



def fine_tuning():
    # ==== Soon ====
    return

def original_model():
    # ==== Soon ====
    return




