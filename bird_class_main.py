import torch
import torchvision

import pandas as pd

import os

import sys

from bird_class_modules import *
from bird_class_cnn_model import *

assert len(sys.argv) > 1, "please provide a model name"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    batch_size = 64
    num_classes = 200
    learning_rate = 0.01
    num_epochs = 50
    workers = 16
    
    # Training dataset
    dataset = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=all_transforms
        )

    train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                               batch_size = batch_size,
                                               shuffle = True, 
                                               num_workers=workers)
    # Load eval dataset
    dataset_eval = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=predict_transform,
        train=False
        )
    
    eval_loader = torch.utils.data.DataLoader(dataset = dataset_eval,
                                               batch_size = batch_size,
                                               shuffle = False, 
                                               num_workers=workers)
    
    model = ConvNeuralNet(num_classes).to(device)
    print(model)
    
    # Set Loss function
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Set scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    #total_step = len(train_loader)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backpropagate error
            loss.backward()

            # Update weights
            optimizer.step()
            train_loss += loss.item()
        
        #scheduler.step()
    
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate(model, eval_loader, criterion, device)
            train_loss_avg = train_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    torch.save(model.state_dict(), sys.argv[1])

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