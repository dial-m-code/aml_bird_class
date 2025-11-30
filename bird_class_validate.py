import torch
import torchvision

import pandas as pd

import os

import sys

from bird_class_modules import *
from bird_class_cnn_model import *

assert len(sys.argv) > 1, "please provide a model path"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    batch_size = 64
    num_classes = 200
    workers = 16

    model = LargeCNN(num_classes)
    model.load_state_dict(torch.load(sys.argv[1], weights_only=True))
    model.eval()
    model.to(device)
    print(model)
    
    # Load eval dataset
    dataset_eval = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=predict_transform,
        train=False,
        use_all= True
        )
    
    eval_loader = torch.utils.data.DataLoader(dataset = dataset_eval,
                                               batch_size = batch_size,
                                               shuffle = False, 
                                               num_workers=workers)
    
    criterion = nn.CrossEntropyLoss()
    
    val_loss, val_acc = validate(model, eval_loader, criterion, device)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / total
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

if __name__ == "__main__":
    main()