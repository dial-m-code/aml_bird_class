import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from skimage import io, transform

import pandas as pd

import os

from PIL import Image

import sys

assert len(sys.argv) > 1, "please provide a model name"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, transform=None, pred=False):
        self.df = pd.read_csv(csv_path)
        
        self.paths = self.df["image_path"].tolist()
        self.labels = (self.df["label"] - 1).tolist()

        self.image_root = image_root
        self.transform = transform

        self.pred = pred
        if self.pred:
            self.img_ids = self.df["id"].tolist()
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        full_path = os.path.join(self.image_root, self.paths[idx][1:])

        img = Image.open(full_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)
        if self.pred:
            return img, label, self.img_ids[idx]
        else:
            return img, label


class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu4 = nn.ReLU()
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(53824, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.relu1(self.conv_layer1(x))
        out = self.relu2(self.conv_layer2(out))
        out = self.max_pool1(out)
        
        out = self.relu3(self.conv_layer3(out))
        out = self.relu4(self.conv_layer4(out))
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.fc2(out)
        return out

#Mean: tensor([0.4859, 0.5032, 0.4440])
#Std: tensor([0.1743, 0.1736, 0.1860])

all_transforms = transforms.Compose([transforms.Resize((128,128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                            mean=[0.4859, 0.5032, 0.4440],
                                            std=[0.1743, 0.1736, 0.1860])
                                         ])

all_transforms_aug = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4859, 0.5032, 0.4440],
        std=[0.1743, 0.1736, 0.1860])
])

def main():
    batch_size = 128
    num_classes = 200
    learning_rate = 0.004
    num_epochs = 50
    
    # Training dataset
    dataset = BirdDataset(
        csv_path="dataset/train_images.csv",
        image_root="dataset/train_images",
        transform=all_transforms_aug
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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
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
    
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

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