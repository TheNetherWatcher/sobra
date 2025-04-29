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
from dataloader import ImageTextDataset

df = pd.read_csv(caption_path)
   
# Training
csv_path = caption_path
dataset = ImageTextDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = CrossAttentionClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



epochs = epochs
losses = []
accuracies = []

for epoch in range(epochs):
    total_loss, correct, total = 0, 0, 0

    for img, text, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(img, text).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Plot loss and accuracy
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(1, epochs+1), losses, 'g-', label='Loss')
ax2.plot(range(1, epochs+1), accuracies, 'b-', label='Accuracy')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Accuracy', color='b')
plt.title('Loss & Accuracy vs. Epoch')
fig.legend(loc='upper right')

# Save the figure
save_path = graph_path
plt.savefig(save_path)

plt.show()

# Save the model weights
torch.save(model.state_dict(), weight_dir)
