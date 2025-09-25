import torchvision
import os
import torch
import data_loader as dt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2
from typing import Tuple

NUM_WORKERS = os.cpu_count()

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
    

def get_data(
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
        train_dir, test_dir= dt.data_from_url(
            url=data_source,
            data_path=data_path,
            save_path=save_path,
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

        transform = weight.transforms()

        train_dataset = data_source(
            root=data_path,
            train=True,
            transform=transform,
            download=True, 
        )

        test_dataset = data_source(
            root=data_path,
            train=False,
            transform=transform,
            download=True, 
        )

        n_classes = train_dataset.classes

        train_dataloader = DataLoader(
            train_dataset,
            batch_size,
            True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            train_dataset,
            batch_size,
            True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

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

