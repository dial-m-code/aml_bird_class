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

#Activation maps

import os
import matplotlib.pyplot as plt

save_dir = "activation_maps"
os.makedirs(save_dir, exist_ok=True)

activations = {}

def get_activation(name):
    def hook(model, inp, out):
        activations[name] = out.detach()
    return hook

# Register hooks  
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))
model.conv5.register_forward_hook(get_activation('conv5'))
model.conv8.register_forward_hook(get_activation('conv8'))

# Forward pass  
sample_img, _, _ = next(iter(pred_loader))
sample_img = sample_img.to(device)
output,_ = model(sample_img)
pred_class = output.argmax(dim=1)

def save_activation_maps(name, fmap, max_channels=6):
    fmap = fmap[0]  # first image in batch
    C = min(max_channels, fmap.shape[0])
    for i in range(C):
        plt.figure(figsize=(3,3))
        plt.imshow(fmap[i].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"{name}_ch{i}")
        plt.savefig(f"{save_dir}/{name}_ch{i}.png", bbox_inches='tight')
        plt.close()

 print("Saving Activation Maps...")
for name, fmap in activations.items():
    save_activation_maps(name, fmap)

#GRAD CAM

import cv2
import numpy as np

#Last Conv2d layer
last_conv = None
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        last_conv = m
print("Grad-CAM Using:", last_conv)

act_store, grad_store = {}, {}

def fwd_hook(module, inp, out):
    act_store['value'] = out

def bwd_hook(module, grad_in, grad_out):
    grad_store['value'] = grad_out[0]

last_conv.register_forward_hook(fwd_hook)
last_conv.register_full_backward_hook(bwd_hook)  # recommended over backward_hook

sample_img2, _, _ = next(iter(pred_loader))
sample_img2 = sample_img2.to(device)

out2, _ = model(sample_img2)

pred2 = out2.argmax(dim=1)

pred2_idx = pred2[-1].item() # Use last image in batch


model.zero_grad()
out2[-1, pred2_idx].backward()

fm = act_store['value'][-1]   # shape (C,H,W) # Extract feature map and gradients for last image
gr = grad_store['value'][-1]  # shape (C,H,W)

weights = gr.mean(dim=(1,2), keepdim=True) # Compute Grad-CAM
cam = torch.relu((weights * fm).sum(dim=0))
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = cam.detach().cpu().numpy()
cam = cv2.resize(cam, (sample_img2.shape[3], sample_img2.shape[2]))

orig = sample_img2[-1].permute(1,2,0).cpu().numpy() # Overlay heatmap on original image
orig = (orig - orig.min()) / (orig.max() - orig.min())
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(np.uint8(orig*255), 0.6, heatmap, 0.4, 0)

plt.imsave(f"{save_dir}/gradcam_heatmap.png", cam, cmap='jet')
plt.imsave(f"{save_dir}/gradcam_overlay.png", overlay)

 