import os

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

# Transforms

# Mean: tensor([0.4859, 0.5032, 0.4440])
# Std: tensor([0.1743, 0.1736, 0.1860])
mean = [0.4859, 0.5032, 0.4440]
std = [0.1743, 0.1736, 0.1860]

all_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.2)
])

# Less agressive
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.1)
])

predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean, std)
])


# Dataset Class

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, transform=None, pred=False, train=True, val_split=0.2, use_all=False):
        if not pred and not use_all:
            df = pd.read_csv(csv_path)
            
            train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            stratify=df["label"],
            random_state=112
            )

            self.df = train_df if train else val_df
            self.df = self.df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
        
        self.paths = self.df["image_path"].tolist()
        self.labels = (self.df["label"] - 1).tolist()

        self.image_root = image_root
        self.transform = transform

        self.pred = pred
        if self.pred:
            self.img_ids = self.df["id"].tolist()
        print(f"Dataset loaded: {len(self.labels)} items.")
        print(f"Mode: {'Prediction' if pred else 'Train/Val'}, Training: {train}, use all: {use_all}")
    
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