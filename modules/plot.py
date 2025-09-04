import torch 
import matplotlib.pyplot as plt
import random
import numpy as np
import data_setup as dt
import seaborn as sns

from torch import nn
from typing import Tuple, List
from PIL import Image
from torchvision.transforms import v2
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

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

        transformed_image = images_transform(pre_images).unsqueeze(0)
        logits = model(transformed_image.to(device))
        preds = torch.softmax(logits, dim=1)
        label = torch.argmax(preds, dim=1)
    
    plt.figure()
    plt.imshow(pre_images)
    plt.title(f"Predicted: {classes[label]} | Probability: {preds.max().cpu().item():.3f}")
    plt.axis('off')

def dataset_prediction(
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
    preds_score = []
    
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
            preds_score.append(preds.max().cpu().item())

    num_classes = len(classes)
    y_true = torch.tensor(true_labels)
    y_pred = torch.tensor(pred_labels)

    f1 = MulticlassF1Score(num_classes=num_classes, average="macro")(y_pred, y_true)
    precision = MulticlassPrecision(num_classes=num_classes, average="macro")(y_pred, y_true)
    recall = MulticlassRecall(num_classes=num_classes, average="macro")(y_pred, y_true)

    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    
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
            plt.title(f'Pred: {pred_label_name} | {preds_score[i]:.3f}\nTrue: {true_label_name}', 
                     c='g', fontsize=10)
        else:
            plt.title(f'Pred: {pred_label_name} | {preds_score[i]:.3f}\nTrue: {true_label_name}', 
                     c='r', fontsize=10)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusionmatrix(
        model: nn.Module, 
        dataset: torch.utils.data.DataLoader,
        classes: List[str],
        device: str
):
    confmat = ConfusionMatrix(task='multiclass', num_classes=len(classes)).to(device)

    model.eval()
    with torch.inference_mode():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            confmat.update(preds, y)

    cm = confmat.compute().cpu()

    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Model {model.__class__.__name__}')
    plt.show()
