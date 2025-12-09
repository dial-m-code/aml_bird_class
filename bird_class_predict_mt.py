"""
Usage: python cnn_bird_test_pred.py [MODEL] [CSV-FILE NAME]
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform
import pandas as pd
import os
from PIL import Image
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from bird_class_modules import *
from bird_class_cnn_model import *

assert len(sys.argv) > 2, "please provide a model and a filename"

num_classes = 200

# load model

#model = ConvNeuralNet(num_classes)
#model = SimpleCNN(num_classes)
#model = LargeCNN(num_classes)
#model = LargeCNN_MT()
model = MediumCNN_MT()
model.load_state_dict(torch.load(sys.argv[1], weights_only=True, map_location="cpu"))
model.to(device)
model.eval()

# load dataset

pred_dataset = BirdDataset(
        csv_path="dataset/test_images_path.csv",
        image_root="dataset/test_images",
        transform=predict_transform,
        pred = True
        )

pred_loader = torch.utils.data.DataLoader(
    dataset=pred_dataset,
    batch_size=16,
    shuffle=False
)

predictions = []
with torch.no_grad():
    for images, _, ids in pred_loader:
        images = images.to(device)

        outputs, _ = model(images)
        predicted = outputs.argmax(dim=1)

        predicted = predicted.cpu().tolist()
        ids = ids.cpu().tolist()
        
        predicted = [p + 1 for p in predicted]

        batch_preds = list(zip(ids, predicted))
        predictions.extend(batch_preds)

df_preds = pd.DataFrame(predictions, columns=["id", "label"])
print(df_preds.head())

df_preds.to_csv(sys.argv[2], index=False)