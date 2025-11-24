import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform
import pandas as pd
import os
from PIL import Image

from cnn_bird_test import ConvNeuralNet, all_transforms, BirdDataset, device

num_classes = 200

# load model

model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load("bird_class_v1_aug_relu", weights_only=True))
model.eval()
model.to(device)

# load dataset

pred_dataset = BirdDataset(
        csv_path="dataset/test_images_path.csv",
        image_root="dataset/test_images",
        transform=all_transforms,
        pred = True
        )

pred_loader = torch.utils.data.DataLoader(
    dataset=pred_dataset,
    batch_size=64,
    shuffle=False
)

predictions = []
with torch.no_grad():
    for images, _, ids in pred_loader:
        images = images.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        predicted = predicted.cpu().tolist()
        ids = ids.cpu().tolist()
        
        predicted = [p + 1 for p in predicted]

        batch_preds = list(zip(ids, predicted))
        predictions.extend(batch_preds)

df_preds = pd.DataFrame(predictions, columns=["id", "label"])
print(df_preds.head())

df_preds.to_csv("predictions_v1_aug_relu.csv", index=False)