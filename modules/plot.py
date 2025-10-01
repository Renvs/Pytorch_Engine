import torch 
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
import timeit 
import pandas as pd

from timeit import default_timer as timer
from torch import nn
from typing import Tuple, List, Dict
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

    start_time = timer()

    with torch.inference_mode():

        transformed_image = images_transform(pre_images).unsqueeze(0)
        logits = model(transformed_image.to(device))
        preds = torch.softmax(logits, dim=1)
        label = torch.argmax(preds, dim=1)

    end_time = timer()
    print(f"Time taken for prediction: {end_time - start_time:.4f} seconds")
    
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

    images = []
    true_labels = []
    pred_labels = []
    preds_score = []
    total_time = []
    
    model.to(device)
    model.eval()

    images_counter = 0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.inference_mode():
        for batch_images, batch_labels in test_data:

            if device =='cuda' and torch.cuda.is_available():
                batch_images = batch_images.to(device)
            else:
                batch_images = batch_images.to('cpu')
            
            start_time = timer()
            
            logits = model(batch_images)
            preds = torch.softmax(logits, dim=1)
            pred_label_batch = torch.argmax(preds, dim=1)
            pred_scores_batch = torch.max(preds, dim=1).values

            end_time = timer()
            batch_time = end_time - start_time
            avg_image_time = batch_time / batch_images.size(0)

            batch_images = batch_images.cpu()

            for i in range(batch_images.size(0)):
                if images_counter >= images_num:
                    break
                
                images.append(batch_images[i])
                true_labels.append(batch_labels[i].item())
                pred_labels.append(pred_label_batch[i].item())
                preds_score.append(pred_scores_batch[i].item())
                total_time.append(avg_image_time)

                images_counter += 1
            
            if images_counter >= images_num:
                break

            
    if images_counter < images_num:
        print(f"Warning: Only {images_counter} images were found in the dataset. Requested {images_num}.")
        images_num = images_counter

    avg_time = np.mean(total_time)
    prediction_time = np.sum(total_time)
    prediction_time_image_dt = pd.DataFrame(total_time, columns=[f'Prediction Time Using {device}'])

    print(f"Average prediction time per image: {avg_time:.4f} seconds |\nTotal prediction time for {images_num} images: {prediction_time:.4f} seconds using {device}")

    num_classes = len(classes)
    y_true = torch.tensor(true_labels)
    y_pred = torch.tensor(pred_labels)

    f1 = MulticlassF1Score(num_classes=num_classes, average="micro")(y_pred, y_true)
    precision = MulticlassPrecision(num_classes=num_classes, average="micro")(y_pred, y_true)
    recall = MulticlassRecall(num_classes=num_classes, average="micro")(y_pred, y_true)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    plt.figure(figsize=(15, 10))
    
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
        
        image = images[i]

        if image.dim() == 3 and image.shape[0] == 3:

            image = image * std + mean
            image = torch.clamp(image, 0, 1)  
            plt.imshow(image.permute(1, 2, 0))

        elif image.dim() == 3 and image.shape[0] == 1:

            plt.imshow(image.squeeze(0), cmap='gray')

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

    return prediction_time_image_dt


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

def plot_dataset(
        dataset: torch.utils.data.DataLoader,
        classes: List[str],
        n_images: int,
): 
    
    all_images = []
    all_labels = []

    with torch.inference_mode():
        for batch_images, batch_labels in dataset:
            for i in range(batch_images.size(0)):
                all_images.append(batch_images[i].cpu())
                all_labels.append(batch_labels[i].item())

            if len(all_images) >= n_images:
                break

    if n_images > len(all_images):
        print(f"Warning: Requested {n_images} images but dataset only has {len(all_images)}. Using all available images.")
        n_images = len(all_images)
    
    sample = random.sample(range(len(all_images)), n_images)

    if n_images <= 4:
        nrows, ncols = 2, 2
    elif n_images <= 9:
        nrows, ncols = 3, 3
    elif n_images <= 16:
        nrows, ncols = 4, 4
    else:
        import math
        ncols = 8
        nrows = math.ceil(n_images / ncols)

    plt.figure(figsize=(ncols * 2, nrows * 2))

    for plot_idx, data_idx in enumerate(sample, 1): 
        plt.subplot(nrows, ncols, plot_idx)
        
        image = all_images[data_idx]
        if image.shape[0] == 3:  
            image = image.permute(1, 2, 0)
        
        plt.imshow(image)
        plt.title(classes[all_labels[data_idx]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def patches_converter(
        dataloader: torch.utils.data.DataLoader,
        num_patches: int,
        patch_size: int
):
    random_sample = random.randint(0, len(dataloader) - 1)
    for i,(image_batch, label_batch) in enumerate(dataloader):
        if i == random_sample:
            random_image = random.randint(0, image_batch.shape[0] - 1)
            image = image_batch[random_image].cpu().permute(1, 2, 0)

    n_patches = num_patches // patch_size
    
    fig, axs = plt.subplots(nrows=n_patches,
                            ncols=n_patches,
                            figsize=(n_patches, n_patches),
                            sharex=True,
                            sharey=True)
    plt.title(f'Patches of Images')
    
    for i in range(n_patches):
        for j in range(n_patches):
            patch_start_y = i * patch_size
            patch_start_x = j * patch_size

            axs[i, j].imshow(image[patch_start_y : patch_start_y + patch_size,
                                        patch_start_x : patch_start_x + patch_size])
            axs[i, j].set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

def plot_loss_curves(results: Dict[str, List[float]], model_name: str):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): A dictionary containing lists of training and testing
                        loss and accuracy metrics.
                        Expected keys: 'train_loss', 'train_acc',
                                       'test_loss', 'test_acc'.
    """
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    ax1.plot(epochs, train_loss, label='train_loss')
    ax1.plot(epochs, test_loss, label='test_loss')
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()

    ax2.plot(epochs, train_accuracy, label='train_accuracy')
    ax2.plot(epochs, test_accuracy, label='test_accuracy')
    ax2.set_title('Accuracy Curves', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()

    fig.suptitle('Training Results ', fontsize=16, fontweight='bold')
    fig.suptitle(f'{model_name}', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
