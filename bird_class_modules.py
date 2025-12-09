import os

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

import numpy as np

# Transforms

# Mean: tensor([0.4859, 0.5032, 0.4440])
# Std: tensor([0.1743, 0.1736, 0.1860])
mean = [0.4859, 0.5032, 0.4440]
std = [0.1743, 0.1736, 0.1860]
"""
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
"""
# Less agressive
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), # was 10
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3)]), p=0.2),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25)
])

predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean, std)
])


# Dataset Class
RS = 90
SPLIT = 0.1

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, transform=None, pred=False, train=True, use_all=False):
        if not pred and not use_all:
            df = pd.read_csv(csv_path)
            
            train_df, val_df = train_test_split(
            df,
            test_size=SPLIT,
            stratify=df["label"],
            random_state=RS
            )

            self.df = train_df if train else val_df
            self.df = self.df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
        
        self.paths = self.df["image_path"].tolist()
        self.labels = (self.df["label"] - 1).tolist()
        self.img_ids = []

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

class BirdDatasetMT(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, attributes,transform=None, pred=False, train=True, use_all=False):
        if not pred and not use_all:
            df = pd.read_csv(csv_path)
            
            train_df, val_df = train_test_split(
            df,
            test_size=SPLIT,
            stratify=df["label"],
            random_state=RS
            )

            self.df = train_df if train else val_df
            self.df = self.df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
        
        self.paths = self.df["image_path"].tolist()
        self.labels = (self.df["label"] - 1).tolist()

        self.attributes = np.load(attributes, allow_pickle=True)

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
            return img, label, self.attributes[label,:]