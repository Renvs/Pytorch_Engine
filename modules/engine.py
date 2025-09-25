import torch 
import os
import torchvision
import torchvision.models as models
import modules.preprocessing as preprocessing, modules.data_loader as data_loader, train_test_step, create_summary, helper
import copy

from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
from torch import nn, optim
from typing import Tuple, Dict, List

NUM_WORKERS = os.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_tracking(
        # ==== Model Params ====
        model: nn.Module,
        model_name: str,
        classifier_name: str,
        weight,

        # ==== Data Params ====
        file_name: str, 
        data_path: str, 
        save_path: str,
        data_source: str or torchvision.datasets.VisionDataset, 
        batch_size: int,
        img_size: Tuple[int, int],

        # ==== Training Params ====
        loss_fn: nn.Module,
        optimizer_class,
        learning_rate:float,
        w_decay:float,
        accuracy,
        writer: SummaryWriter,
        patience: int,
        device: str = device,
        num_workers: int = NUM_WORKERS,
        epochs: int = 10,
        warm_epochs: int = 0
):  

    # ==== Get The Data & Preprocessing The Data ====

    print('\n[INFO] Preparing Dataloader\n')

    train_dataloader, test_dataloader, class_names = preprocessing.get_dataloader(
        data_source=data_source,
        weight=weight,
        batch_size=batch_size,
        image_size=img_size,
        data_path=data_path, 
        save_path=save_path,
        filename=file_name,
        num_workers=num_workers
    )

    # ==== Prep The Model ====

    print('\n[INFO] Preparing Model Classifier\n')

    model = model.to(device)

    for params in model.parameters():
        params.requires_grad = False

    original_classifier = helper.get_nested_attr(model, classifier_name)

    in_features = None

    is_conv_layer = False

    if isinstance(original_classifier, nn.Linear):
        in_features = original_classifier.in_features

    elif isinstance(original_classifier, nn.Conv2d):
        in_features = original_classifier.in_channels
        is_conv_layer = True

    else: 
        for layer in reversed(list(original_classifier.modules())):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break

            elif isinstance(layer, nn.Conv2d):
                in_features = layer.in_channels
                is_conv_layer = True
                break

    if in_features is None:
        raise ValueError(f"Could not find Linear layer named '{classifier_name}' in the model")

    if is_conv_layer:
        new_classifier = nn.Conv2d(in_channels=in_features, out_channels=len(class_names), kernel_size=(1, 1), stride=(1, 1))
    else:
        new_classifier = nn.Linear(in_features=in_features, out_features=len(class_names))

    helper.set_nested_attr(model, classifier_name, new_classifier)

    new_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=new_params, lr=learning_rate, weight_decay=w_decay)

    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs - warm_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warm_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warm_epochs]
    )

    # ==== Dummy Pass The Model

    dummy_test = summary(
        model, 
        input_size= (1, 3, img_size[0], img_size[1]),
        col_names= ['input_size', 'output_size', 'num_params', 'trainable'],
        col_width= 20,
        row_settings=['var_names']
    )

    print(dummy_test)

    accuracy = accuracy.to(device)
        
    # ==== Train The Model ====

    result = train_test_step.summary_writer_addon(
        model=model, 
        model_name=model_name, 
        model_path=save_path,
        train_data=train_dataloader,
        test_data=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        accuracy=accuracy,
        epochs=epochs,
        batch_size=batch_size,
        image_size=img_size[0],
        writer=writer,
        device=device,
        patience=patience
    )

    return result

def multiple_tracking(
  model_list: Dict[str, Tuple[callable, str, str]],
  img_size:Tuple[int, int],
  data_list: Dict[str, str],
  epochs: Dict[int, int],
  batch_size: int,
  optimizer_class,
  learning_rate: float, 
  w_decay: float,
  accuracy_fn,
  num_workers: int,
  save_path: str,    
  model_path: str,
  patience: int,
  device: str = device ,
):
    
    experiment_num = 0
    result = []
    best_loss = float('inf')

    total_experiment = len(model_list)*len(data_list)*len(epochs)

    for model_name, (model_fn, weight, classifier) in model_list.items():
        for data_name, data_source in data_list.items():
            for epoch, warmup_epoch in epochs.items():
                    
                experiment_num += 1

                print(f'[INFO] \nSTARTING EXPERIMENT {experiment_num}/{total_experiment}')
                print(f'[INDO] MODEL: {model_name}, DATA: {data_name}, \nEPOCHS: {epoch}\n')

                model = model_fn(weights=weight).to(device)
                
                writer = create_summary.create_summary_writer(
                    f'{model_name}', f'{epoch}_epoch', f'{data_name}'
                )

                experiment_result = single_tracking(
                    model=model,
                    model_name=model_name,
                    classifier_name=classifier,
                    weight=weight,

                    file_name=data_name, 
                    data_path='dataset',
                    save_path=save_path,
                    data_source=data_source, 
                    batch_size=batch_size,
                    img_size=img_size,

                    loss_fn=nn.CrossEntropyLoss(),
                    optimizer_class=optimizer_class,
                    learning_rate=learning_rate, 
                    w_decay=w_decay,
                    accuracy=accuracy_fn,
                    writer=writer,
                    patience=patience,
                    device=device,
                    num_workers=num_workers,
                    epochs=epoch,
                    warm_epochs=warmup_epoch
                )

                result.append(experiment_result)
                
                current_min_loss = min(experiment_result['test_loss'])

                if current_min_loss < best_loss:
                    best_loss = current_min_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    print(f'Save best weights with loss: {current_min_loss:.4f}')

            data_loader.save_models(model, save_path, model_name)

    if best_weights is not None:
        data_loader.save_models(model, save_path, 'best_model.pt')
    else:
        print('No model to save')

    return result
    
def fine_tuning():
    # ==== Soon ====
    return

def original_model():
    # ==== Soon ====
    return

# def feature_extraction(model: nn.Module,
#                        model_name: str,
#                        classifier_name: str, 
#                        file_name: str, 
#                        data_path: str, 
#                        save_path: str,
#                        url: str, 
#                        weight: models,
#                        batch_size: int,
#                        loss_fn: nn.Module,
#                        optimizer,
#                        learning_rate: float, 
#                        w_decay: float,
#                        patience:int,
#                        img_size: Tuple[int, int],
#                        accuracy,
#                        device: str = device,
#                        num_workers: int = NUM_WORKERS,
#                        warm_epochs: int = 4,
#                        epochs: int = 10,
#                        ):
    
#     # ==== Get The Data ====

#     print('[INFO] Getting Data')
#     train_dir, test_dir, model_path = get_data.get_data(
#         url= url, data_path= data_path, save_path= save_path, filename= file_name
#     )

#     # ==== Prep The Data ====

#     print('[INFO] Preparing Dataloader')
#     train_dataloader, test_dataloader, class_names = preprocessing.load_dataloader(
#         train_dir, test_dir, batch_size, weight, img_size, num_workers
#     )

#     # ==== Prep The Model ====

#     print('[INFO] Preparing Model Classifier')

#     model = model.to(device)

#     for params in model.parameters():
#         params.requires_grad = False

#     original_classifier = helper.get_nested_attr(model, classifier_name)
#     in_features = None
#     is_conv_layer = False

#     if isinstance(original_classifier, nn.Linear):
#         in_features = original_classifier.in_features
#     elif isinstance(original_classifier, nn.Conv2d):
#         in_features = original_classifier.in_channels
#         is_conv_layer = True
#     else: 
#         for layer in reversed(list(original_classifier.modules())):
#             if isinstance(layer, nn.Linear):
#                 in_features = layer.in_features
#                 break
#             elif isinstance(layer, nn.Conv2d):
#                 in_features = layer.in_channels
#                 is_conv_layer = True
#                 break

#     if in_features is None:
#         raise ValueError(f"Could not find Linear layer named '{classifier_name}' in the model")

#     if is_conv_layer:
#         new_classifier = nn.Conv2d(in_channels=in_features, out_channels=len(class_names), kernel_size=(1, 1), stride=(1, 1))
#     else:
#         new_classifier = nn.Linear(in_features=in_features, out_features=len(class_names))

#     helper.set_nested_attr(model, classifier_name, new_classifier)

#     new_params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optimizer(params=new_params, lr=learning_rate, weight_decay=w_decay)

#     main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs - warm_epochs)
#     warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warm_epochs)
#     scheduler = optim.lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[warmup_scheduler, main_scheduler],
#         milestones=[warm_epochs]
#     )

#     # ==== Dummy Pass The Model

#     dummy_test = summary(
#         model, 
#         input_size= (1, 3, img_size[0], img_size[1]),
#         col_names= ['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width= 20,
#         row_settings=['var_names']
#     )

#     print(dummy_test)

#     optimizer = optimizer(model.parameters(), lr=learning_rate)
#     accuracy = accuracy.to(device)
        
#     result = train_test_step.train(
#         model=model, 
#         train_data=train_dataloader,
#         test_data=test_dataloader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         accuracy=accuracy,
#         model_path=model_path,
#         epochs=epochs,
#         device=device,
#         patience=patience
#     )

#     return result