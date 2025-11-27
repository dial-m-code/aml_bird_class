import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = BirdDataset(
    csv_path="dataset/train_images.csv",
    image_root="dataset/train_images",
    transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        # images: (B, C, H, W)
        batch_samples = images.size(0)
    
        # flatten
        images = images.view(batch_samples, images.size(1), -1)
    
        # mean and std per channel
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    # Final mean and std
    mean /= total_images
    std /= total_images
    
    print("Mean:", mean)
    print("Std:", std)

class BirdDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        
        self.paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()

        self.image_root = image_root
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        full_path = os.path.join(self.image_root, self.paths[idx][1:])

        img = Image.open(full_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == "__main__":
    main()