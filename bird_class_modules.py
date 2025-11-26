import os

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image

# Transforms

# Mean: tensor([0.4859, 0.5032, 0.4440])
# Std: tensor([0.1743, 0.1736, 0.1860])


all_transforms = transforms.Compose([
    transforms.CenterCrop(500),
    #transforms.RandomResizedCrop(128),
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(
        mean=[0.4859, 0.5032, 0.4440],
        std=[0.1743, 0.1736, 0.1860]
    ),
])

predict_transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize((128, 128)),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),  # convert to tensor
    transforms.Normalize(
        mean=[0.4859, 0.5032, 0.4440],
        std=[0.1743, 0.1736, 0.1860]
    ),
])

# Dataset Class

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