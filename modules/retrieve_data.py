import torchvision
import os
import torch
import inspect
import torch.nn as nn
import zipfile
import requests

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2
from typing import Tuple
from pathlib import Path

NUM_WORKERS = os.cpu_count()

def get_data(url: str, data_path: str, filename: str = None):
    try:

        if data_path and filename:
            print(f'[INFO] Checking Directory at {data_path}')
            data_path = Path(data_path)
            data_path.mkdir(parents=True, exist_ok=True)
            file_path = data_path / filename

            if url:
                with open(file_path.with_suffix('.zip'), 'wb') as f:        
                    print(f'[INFO] Downloading from {url} at {file_path}')
                    request = requests.get(url, stream=True)
                    f.write(request.content)
            else: 
                raise ValueError('[ERROR] No URL provided')

            with zipfile.ZipFile(file_path.with_suffix('.zip'), 'r') as zip_file:
                print(f'[INFO] Extracting from {file_path}')
                zip_file.extractall(file_path)
            
            if os.path.exists(file_path.with_suffix('.zip')):
                os.remove(file_path.with_suffix('.zip'))

        else:
            raise ValueError('[ERROR] No data path / filename provided')

    except ValueError as e:
        print(e)

    train_dir = file_path / 'train'
    test_dir = file_path / 'test'

    return train_dir, test_dir

def save_models(model: nn.Module, save_path: str, model_name: str):
    try:
        if save_path:
                print(f'[INFO] Created Directory at {save_path}')
                target_path = Path(save_path)
                target_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError('[ERROR] No save path provided')

        model_path = target_path / f'{model_name}.pt'

        torch.save(model.state_dict(), model_path)
        print(f'[INFO] Model saved at {model_path} / {model_name}.pt')

    except Exception as e:        
        print(f"[ERROR] Failed to save model: {e}")

def dataloader_from_zip(
        train_dir:str,
        test_dir:str, 
        batch_size:int, 
        num_workers: int,
        image_size: Tuple[int, int] = None,
        weight: torchvision.models = None
):
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=image_size),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])

    if weight:
        transform = weight.transforms()

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_data, 
        batch_size, 
        True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data, 
        batch_size, 
        False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    n_classes = train_data.classes

    return train_dataloader, test_dataloader, n_classes
    

def create_dataloader(
        data_source: str or Callable,
        batch_size: int, 
        image_size: Tuple[int, int],
        data_path: str, 
        save_path: str,
        filename: str,
        weight = None,
        num_workers: int = 0
):  
    if isinstance(data_source, str):

        print(f'[INFO] Getting data from URL: {data_source} ')

        train_dir, test_dir = get_data(
            url=data_source,
            data_path=data_path,
            filename=filename
        )

        return dataloader_from_zip(
            train_dir=train_dir,
            test_dir=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            weight=weight
        )
    
    elif issubclass(data_source, VisionDataset):

        print(f'\n[INFO] Getting data from torchvision datasets: {data_source.__name__}')

        sig = inspect.signature(data_source.__init__)
        transform = weight.transforms()
    
        dataset_params = {
            'root': data_path,
            'transform': transform,
            'download': True,
            'target_transform': None
        }

        if 'train' in sig.parameters:

            train_dataset = data_source(**{**dataset_params, 'train': True})
            test_dataset = data_source(**{**dataset_params, 'train': False})

        elif 'split' in sig.parameters:

            train_dataset = data_source(**{**dataset_params, 'split': 'train'})
            test_dataset = data_source(**{**dataset_params, 'split': 'test'})
            
        else:

            raise ValueError('[ERROR] The dataset class does not have the expected parameters (split or train)')

        n_classes = train_dataset.classes

        dataloader_params ={
            'batch_size': batch_size,
            'shuffle': bool,
            'num_workers': num_workers,
            'pin_memory': True
        }

        if train_dataset and test_dataset:

            train_dataloader = DataLoader(train_dataset, {**dataloader_params, 'shuffle': True})
            test_dataloader = DataLoader(test_dataset, {**dataloader_params, 'shuffle': False})

        return train_dataloader, test_dataloader, n_classes
    else:
        raise ValueError('[ERROR] data_source must be a URL string or a torchvision dataset class')

def load_images(
        test_dir: str,
        image_size: Tuple[int, int]
        ):

    transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=image_size),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])
    
    images = datasets.ImageFolder(test_dir, transform=transforms)
    class_names = images.classes

    return images, class_names

