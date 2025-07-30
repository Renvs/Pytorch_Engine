import torch 
import os
import torchvision.models as models

from torchinfo import summary
from torch import nn, optim
from modules import data_setup, get_data, train_test_step

NUM_WORKERS = os.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def feature_extraction(model: nn.Module,
                       file_name: str, 
                       data_path: str, 
                       save_path: str,
                       url: str, 
                       weight: models,
                       batch_size: int,
                       loss_fn: nn.Module,
                       optimizer: optim.Optimizer,
                       accuracy,
                       device: str = device,
                       num_workers: int = NUM_WORKERS,
                       epochs: int = 10,
                       ):
    
    # ==== Get The Data ====

    print('[INFO] Getting Data')
    train_dir, test_dir, model_path = get_data.get_data(
        url= url, data_path= data_path, save_path= save_path, filename= file_name
    )

    # ==== Prep The Data ====

    print('[INFO] Preparing Dataloader')
    train_dataloader, test_dataloader, class_names = data_setup.load_dataloader(
        train_dir, test_dir, weight, batch_size, num_workers
    )

    # ==== Prep The Model ====

    print('[INFO] Preparing Model Classifier')

    crop_size = weight.transforms().crop_size[0]

    model = model.to(device)

    original_classifier = model.classifier
    dropout_layer = original_classifier[0]
    dropout_p = dropout_layer.p
    dropout_inplace = dropout_layer.inplace

    for params in model.features.parameters():
        params.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=dropout_inplace),
        nn.Linear(in_features= model.classifier[1].in_features, out_features= len(class_names)),
    )

    print(f'[INFO] {model.classifier[1].in_features}')

    # ==== Dummy Pass The Model

    dummy_test = summary(
        model, 
        input_size= (1, 3, crop_size, crop_size),
        col_names= ['input_size', 'output_size', 'num_params', 'trainable'],
        col_width= 20,
        row_settings=['var_names']
    )

    print(dummy_test)

    best_loss = float('inf')
        
    result = train_test_step.train(
        model=model, 
        train_data=train_dataloader,
        test_data=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy=accuracy,
        epochs=epochs,
        device=device
    )

    return result



def fine_tuning():
    # ==== Soon ====
    return

def original_model():
    # ==== Soon ====
    return