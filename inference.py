from vlm import vlm
from model import CrossAttentionClassifier 
import pandas as pd
import os
import torch
from rag import rag
from typing import Text
from config import device, weight_dir, inference_csv
from dataloader import ImageTextDataset
from torch.utils.data import DataLoader
import pandas as pd

inf_prompt = """Please examine the uploaded image of a soybean leaf and provide a comprehensive description of all visible symptoms. Focus on details such as:

Leaf discoloration patterns: Note any yellowing (chlorosis), browning, or other color changes, specifying whether they occur between veins (interveinal) or along the edges (marginal).

Presence of fungal indicators: Identify any signs of fungal infections, such as white powdery coatings indicative of powdery mildew, grayish fuzz on the underside of leaves suggesting downy mildew, or dark spots that may point to brown spot disease .

Lesion characteristics: Describe any spots or lesions, including their color, shape, and distribution on the leaf surface.
Land-Grant Press

Additional abnormalities: Mention any other notable features such as leaf curling, wilting, or premature defoliation.

Provide a detailed analysis to assist in accurately diagnosing the specific disease affecting the soybean leaf as well as some preventive measures that can be used."""

caption_prompt = """The input is an image of a leaf. Please analyze the image and describe the following:
Leaf Condition:
Any discoloration, such as yellowing, browning, or spotting.
The extent of damage, if visible (e.g., edges curling, holes, or lesions).
Are there signs of wilting or drooping?

Patterns and Texture:
Describe any visible patterns, spots, or markings on the surface of the leaf (e.g., fungal growth, pest damage, or nutrient deficiency signs).
Any distinct textures (e.g., smooth, rough, glossy, etc.).
Is there any visible fungal infection (mold, mildew, etc.)?

Leaf Structure:
The general shape and size of the leaf (e.g., normal, irregular, or distorted).
Are the veins visible? If so, describe their color, pattern, and clarity.

Potential Disease or Pest Infestation:
Identify if there are any symptoms of common diseases (e.g., rust, blight, powdery mildew).
Are there any signs of pest damage (e.g., holes, bite marks, or larvae)?

Environmental Stress Indicators:
Any signs of environmental stress (e.g., sunburn, drought stress, or overwatering).
Does the leaf appear to be affected by external factors (e.g., pollution, wind damage)?

Please provide a detailed analysis, highlighting any unusual features or concerns about the leaf's health."""

model = CrossAttentionClassifier().to(device)
model.load_state_dict(torch.load(weight_dir))
model.eval()

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample.png")

def inference(img_path=file_path):
    caption = vlm(caption_prompt, img_path)
    data = {'image name': [img_path], 'output': [caption], 'label': ['inference']}
    df = pd.DataFrame(data)
    df.to_csv(inference_csv, index=False)
    val_dataset = ImageTextDataset(inference_csv)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    with torch.no_grad():
      for img , text , labels in val_dataloader:
          outputs = model(img , text).squeeze()
    # print(outputs)
    if outputs < 0.5:
        inf_caption = vlm(inf_prompt)
        final_txt = rag(inf_caption)
        return final_txt
    else:
      return "Healthy leaf image"