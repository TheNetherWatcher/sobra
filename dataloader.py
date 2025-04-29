import pandas as pd
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Text
from config import device , caption_path , healthy_path, disease_path , batch_size , learning_rate , epochs , weight_dir , graph_path
from model import extract_image_embedding, extract_text_embedding, CrossAttentionClassifier 
import matplotlib.pyplot as plt

class ImageTextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print(row)
                if row['label'] == "healthy":
                  label = 1
                elif row['label'] == "disease":
                  label = 0
                else:
                  label = 3
                if label == 1:
                  img_path = os.path.join(healthy_path, row['image name'])
                elif label == 0:
                  img_path = os.path.join(disease_path, row['image name'])
                else:
                  img_path = row['image name']
                self.data.append((img_path, row['output'], label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption, label = self.data[idx]
        img_embeddings = extract_image_embedding(img_path).squeeze(0)  # Shape: (197, 768)
        text_embeddings = extract_text_embedding(caption).squeeze(0)  # Shape: (768,)
        return img_embeddings, text_embeddings, torch.tensor(float(label), dtype=torch.float32).to(device)
 