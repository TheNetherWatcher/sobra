import pandas as pd
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Text
from config import device , val_csv_file , weight_dir
from model import extract_image_embedding, extract_text_embedding, CrossAttentionClassifier 
import matplotlib.pyplot as plt
from dataloader import ImageTextDataset
from model import CrossAttentionClassifier

val_dataset = ImageTextDataset(val_csv_file)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = CrossAttentionClassifier().to(device)
model.load_state_dict(torch.load(weight_dir))
model.eval()

total_correct, total_samples = 0, 0
with torch.no_grad():
    for img , text , labels in val_dataloader:
        outputs = model(img , text).squeeze()
        preds = (outputs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

print(f"Validation Accuracy: {total_correct / total_samples:.4f}")