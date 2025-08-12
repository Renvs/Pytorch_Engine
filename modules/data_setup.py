import torchvision
import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from typing import Tuple

NUM_WORKERS = os.cpu_count()

def load_dataloader(train_dir: str, test_dir:str, weight: torchvision.models, batch_size: int, num_workers: int = NUM_WORKERS):

    weights = weight
    transform = weights.transforms()

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes
    
    train_dataloader = DataLoader(train_data, batch_size, True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names

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

