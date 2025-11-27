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
    batch_size = 32
    num_classes = 200
    learning_rate = 0.002
    num_epochs = 100
    
    # Training dataset
    dataset = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=all_transforms
        )

    train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                               batch_size = batch_size,
                                               shuffle = True, 
                                               num_workers=4)
    
    
    model = ConvNeuralNet(num_classes).to(device)
    print(model)
    
    # Set Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005) 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Set scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    
    total_step = len(train_loader)

    model.train()
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Backpropagate error
            loss.backward()

            # Update weights
            optimizer.step()
            #scheduler.step()
    
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    dataset_eval = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=predict_transform
        )
    
    eval_loader = torch.utils.data.DataLoader(dataset = dataset_eval,
                                               batch_size = batch_size,
                                               shuffle = False, 
                                               num_workers=4)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the {} train images: {} %'.format(len(dataset), 100 * correct / total))
    
    torch.save(model.state_dict(), sys.argv[1])

if __name__ == "__main__":
    main()