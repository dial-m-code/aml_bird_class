import torch
import torchvision

import pandas as pd

import os
import sys
import time

from bird_class_modules import *
from bird_class_cnn_model import *

assert len(sys.argv) > 1, "please provide a model name"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
learning_rate = 0.05
num_epochs = 400
weight_decay = 1e-4
workers = 16


def main():
    # Load training dataset
    dataset = BirdDatasetMT(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        attributes="dataset/attributes.npy",
        transform=train_transforms,
        #use_all=True
        )

    train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                               batch_size = batch_size,
                                               shuffle = True, 
                                               num_workers=workers)
    # Load eval dataset
    dataset_eval = BirdDatasetMT(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        attributes="dataset/attributes.npy",
        transform=predict_transform,
        train=False
        )

    eval_loader = torch.utils.data.DataLoader(dataset = dataset_eval,
                                               batch_size = batch_size,
                                               shuffle = False, 
                                               num_workers=workers)
    
    #model = ConvNeuralNet(num_classes).to(device)
    #model = SimpleCNN(num_classes).to(device)
    #model = LargeCNN_MT().to(device)
    model = MediumCNN_MT().to(device)
    print(model)
    
    # Set Loss function
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_class = nn.CrossEntropyLoss()
    criterion_attribute = nn.BCEWithLogitsLoss()
    
    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Set scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) # eta_min=1e-5
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80, 140, 180], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    train_start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (images, labels, attributes) in enumerate(train_loader):  
            # Move tensors to GPU
            images = images.to(device)
            labels = labels.to(device)
            attributes = attributes.to(device)

            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            classes_out, attributes_out = model(images)

            # Loss function
            loss_class = criterion_class(classes_out, labels)
            loss_attribute = criterion_attribute(attributes_out, attributes)
            loss = loss_class + loss_attribute
            
            # Backpropagate error
            loss.backward()

            # Update weights
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
    
        # Validate every 5 epochs
        validate_every_epoch = True
        if (epoch + 1) % 5 == 0 or validate_every_epoch:
            val_loss, val_acc = validate(model, eval_loader, criterion_class, criterion_attribute, device)
            train_loss_avg = train_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Class loss: {loss_class:.4f}, Attribute loss: {loss_attribute:.4f}')
            print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), sys.argv[1])
                print('-new model saved-')
            print('---')
    
    torch.save(model.state_dict(), sys.argv[1]+"_final-epoch")
    duration_train = (time.time()-train_start_time)//60
    print(f"Training took: {duration_train} minutes")
    write_to_log("equal loss weights, different augmentation, more epochs", best_val_acc, val_acc, train_loss_avg, val_loss, duration_train, num_epochs)

def validate(model, val_loader, criterion_class, criterion_attribute, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, attributes in val_loader:
            images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)
            classes_out, attributes_out = model(images)
            loss_class = criterion_class(classes_out, labels)
            loss_attribute = criterion_attribute(attributes_out, attributes)
            loss = loss_class + 0.3 * loss_attribute
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(classes_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / total
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

def write_to_log(description, best_acc, last_acc, train_loss, val_loss, duration, num_epochs):
    with open("train_log.txt", "a") as f:
        f.write("---\n")
        f.write(f"Training description: {description}\n")
        f.write(f"Best Acc.: {best_acc:.2f}%, Last Acc.: {last_acc:.2f}%\n")
        f.write(f"Train loss: {train_loss:.4f}, Val. loss: {val_loss:.4f}\n")
        f.write(f"Training took: {duration} minutes.\n")
        f.write(f"Trained for {num_epochs} epochs.\n")

if __name__ == "__main__":
    main()