import torch 
import os
import torchvision.models as models
import data_setup, get_data, train_test_step, create_summary

from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
from torch import nn, optim
from typing import Tuple, Dict, List

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
                       learning_rate: float, 
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

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    accuracy = accuracy.to(device)
        
    result = train_test_step.train(
        model=model, 
        train_data=train_dataloader,
        test_data=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy=accuracy,
        model_path=model_path,
        epochs=epochs,
        device=device
    )

    return result

def single_tracking(
        model: nn.Module,
        file_name: str, 
        data_path: str, 
        save_path: str,
        url: str, 
        weight,
        batch_size: int,
        loss_fn: nn.Module,
        optimizer,
        learning_rate: float, 
        accuracy,
        writer: SummaryWriter,
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

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    accuracy = accuracy.to(device)
        
    # ==== Train The Model ====

    result = train_test_step.single_tracking(
        model=model, 
        train_data=train_dataloader,
        test_data=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy=accuracy,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        image_size=crop_size,
        writer=writer,
        device=device
    )

    return result

def multiple_tracking(
  model_list: Dict[str, List[str]],
  data_list: Dict[str, List[str]],
  epochs: List[int],
  batch_size: int,
  learning_rate: float, 
  optimizer_fn,
  accuracy_fn,
  num_workers: int,
  save_path: str,    
  model_path: str,
  device: str = device ,
) :
    
    experiment_num = 0
    result = []

    total_experiment = len(model_list)*len(data_list)*len(epochs)

    for model_name, (model_fn, weight) in model_list.items():
        for data_name, url in data_list.items():
            for epoch in epochs:

                experiment_num += 1

                print(f'[INFO] STARTING EXPERIMENT {experiment_num}/{total_experiment}')
                print(f'[INDO] MODEL: {model_name}, DATA: {data_name}, EPOCHS: {epoch}')

                model = model_fn(weights=weight).to(device)
                
                writer = create_summary.create_summary_writer(
                    f'{model_name}', f'{epoch}_epoch', f'{data_name}'
                )

                experiment_result = single_tracking(
                    model=model,
                    file_name=data_name, 
                    data_path='dataset',
                    save_path=save_path,
                    model_path=model_path,
                    url= url, 
                    weight=weight,
                    batch_size=batch_size,
                    loss_fn=nn.CrossEntropyLoss(),
                    optimizer=optimizer_fn,
                    learning_rate=learning_rate, 
                    accuracy=accuracy_fn,
                    writer=writer,
                    device=device,
                    num_workers=num_workers,
                    epochs=epoch
                )

                result.append(experiment_result)
                
                if result['test_loss'] < best_loss:
                    best_loss = result['test_loss']
                    torch.save(model.state_dict(), model_path)
                    print(f'Save at {model_path} Loss: {result['test_loss']:.4f}')

                torch.save(model.state_dict(), f'{save_path}/{model_name}_{data_name}_{epoch}.pt')

    return result
    
def fine_tuning():
    # ==== Soon ====
    return

def original_model():
    # ==== Soon ====
    return