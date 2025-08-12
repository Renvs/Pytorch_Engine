import torch 
import matplotlib.pyplot as plt
import random
import numpy as np
import data_setup as dt

from torch import nn
from typing import Tuple, List
from PIL import Image
from torchvision.transforms import v2

def images_prediction(
    model: nn.Module,
    image: str, 
    image_size: Tuple[int, int],
    classes: List[str],
    device: str,
    transform = None,
):
    
    pre_images = Image.open(image)

    if transform is not None:
        images_transform = transform
    else:
        images_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=image_size),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():

        transformed_image = images_transform(pre_images).unsqueeze(dim=1)
        logits = model(transformed_image.to(device))
        preds = torch.softmax(logits, dim=1)
        label = torch.argmax(preds, dim=1)
    
    plt.figure()
    plt.imshow(pre_images)
    plt.title(f"Predicted: {classes[label]} | Probability: {preds.max().cpu().item():.3f}")

def dataset_prediction(
    model: nn.Module,
    test_data: torch.utils.data.DataLoader,
    classes: List[str],
    images_num: int, 
    device: str,
):
    model.to(device)

    images = []
    labels = []

    for label, image in random.sample(list(test_data), k=images_num):
        images.append(image)
        labels.append(label)

    model.eval()
    with torch.inference_mode():
        for image in images:
            image.unsqueeze(dim=0).to(device)
            logits = model(image.to(device))
            preds = torch.softmax(logits, dim=1)
            label = torch.argmax(preds, dim=1).cpu()
    
    plt.figure(figsize=(15, 10))
    nrows = images_num // 2
    ncols = images_num // 2
    for i, images in enumerate(images):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(images.squeeze(dim=1))

        pred_label = classes[label[i]]
        true_label = classes[labels[i]]
        
    if pred_label == true_label:
        plt.title(f'Pred: {pred_label} | True: {true_label}', c='g', fontsize=10)
    else:
        plt.title(f'Pred: {pred_label} | True: {true_label}', c='r', fontsize=10)

        plt.axis(False)
    