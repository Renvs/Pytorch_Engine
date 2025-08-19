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

def dataset_prediction_v2(
    model: nn.Module,
    test_data: torch.utils.data.DataLoader,
    classes: List[str],
    images_num: int, 
    device: str,
):
    model.to(device)
    model.eval()

    all_images = []
    all_labels = []
    
    with torch.inference_mode():
        for batch_images, batch_labels in test_data:
            for i in range(batch_images.size(0)):
                all_images.append(batch_images[i].cpu())
                all_labels.append(batch_labels[i].item())
    
    if images_num > len(all_images):
        print(f"Warning: Requested {images_num} images but dataset only has {len(all_images)}. Using all available images.")
        images_num = len(all_images)
    
    sampled_indices = random.sample(range(len(all_images)), images_num)
    
    images = []
    true_labels = []
    pred_labels = []
    
    # Make predictions for sampled images
    with torch.inference_mode():
        for idx in sampled_indices:
            image = all_images[idx]
            true_label = all_labels[idx]
            
            # Make prediction
            image_input = image.unsqueeze(0).to(device)
            logits = model(image_input)
            preds = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(preds, dim=1).cpu().item()
            
            images.append(image)
            true_labels.append(true_label)
            pred_labels.append(pred_label)
    
    # Plot results (same plotting code as above)
    plt.figure(figsize=(15, 10))
    
    # Calculate grid dimensions
    if images_num <= 4:
        nrows, ncols = 2, 2
    elif images_num <= 9:
        nrows, ncols = 3, 3
    elif images_num <= 16:
        nrows, ncols = 4, 4
    else:
        import math
        nrows = int(math.sqrt(images_num))
        ncols = math.ceil(images_num / nrows)
    
    for i in range(len(images)):
        plt.subplot(nrows, ncols, i+1)
        
        # Handle different image formats
        image = images[i]
        if image.dim() == 3:
            if image.shape[0] == 1:  # Grayscale
                plt.imshow(image.squeeze(0), cmap='gray')
            elif image.shape[0] == 3:  # RGB
                plt.imshow(image.permute(1, 2, 0))
        else:
            plt.imshow(image, cmap='gray')
        
        pred_label_name = classes[pred_labels[i]]
        true_label_name = classes[true_labels[i]]
        
        if pred_label_name == true_label_name:
            plt.title(f'Pred: {pred_label_name}\nTrue: {true_label_name}', 
                     c='g', fontsize=10)
        else:
            plt.title(f'Pred: {pred_label_name}\nTrue: {true_label_name}', 
                     c='r', fontsize=10)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    