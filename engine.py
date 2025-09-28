import os
import math
import random

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from collections import Counter
from src.data import process
from .src.kmeans_for_clusters import find_clusters
from .src.dataloader import get_loaders
from .src.UNet import UNet

import warnings
warnings.filterwarnings('ignore')

print("Loading the device and data...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(f"Operating on device: {device}")

data_dir, train_dir, val_dir, train_fns, val_fns = process()
label_model = find_clusters(num_colors = 1000, num_classes = 10)

# ---------------------- Global Constants
batch_size = 4
epochs = 10
lr = 0.01
num_classes = 10
# -----------------------

train_loader = get_loaders(batch_size, label_model, train_dir)
print("Training data has been loaded, preparing to train...")

model = UNet(num_classes = num_classes).to(device)

# ---------------------- Optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
# ----------------------

step_losses = []
epoch_losses = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for X, Y in tqdm(train_loader, leave = True):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())

    epoch_losses.append(epoch_loss / len(train_loader))

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)

model_name = "U-Net.pth"
torch.save(model.state_dict(), model_name)

model_path = "U-Net.pth"
model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

test_batch_size = 4
val_loader = get_loaders(batch_size = test_batch_size, dire = val_dir)

X, Y = next(iter(val_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim = 1)
print(Y_pred.shape)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))
])

fig, axes = plt.subplots(test_batch_size, 3, figsize=(3 * 5, test_batch_size * 5))

for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")