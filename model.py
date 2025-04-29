import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from transformers import AutoProcessor , BertModel, BertTokenizer , ViTModel, ViTFeatureExtractor , AutoImageProcessor
from PIL import Image
from config import device


vit_model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTModel.from_pretrained(vit_model_name).eval().to("cuda")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

def extract_image_embedding(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = preprocess(image).unsqueeze(0).to("cuda")
    inputs = image_processor(images=image, return_tensors="pt", padding="max_length", truncation=True).to("cuda")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    last_hidden_states = outputs.last_hidden_state.to(device)
    return last_hidden_states


bert_model = BertModel.from_pretrained("bert-base-uncased").to("cuda").eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def extract_text_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**tokens).last_hidden_state[:, 0, :]
    return outputs.squeeze(0)


class CrossAttentionClassifier(nn.Module):
    def __init__(self, img_dim=768, text_dim=768, num_heads=8, hidden_dim=512):
        super().__init__()

        self.cross_attn_img_to_txt = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_txt_to_img = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(img_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img_embeddings, txt_embeddings):
        # Cross-attention with image as query, text as key/value
        enhanced_img, _ = self.cross_attn_img_to_txt(img_embeddings, txt_embeddings.unsqueeze(1), txt_embeddings.unsqueeze(1))

        # Cross-attention with text as query, image as key/value
        enhanced_txt, _ = self.cross_attn_txt_to_img(txt_embeddings.unsqueeze(1), img_embeddings, img_embeddings)

        # Concatenate along feature dimension
        fused_embedding = torch.cat((enhanced_img.mean(dim=1), enhanced_txt.squeeze(1)), dim=1)

        return self.fc(fused_embedding)