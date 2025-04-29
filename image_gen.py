import torch
from diffusers import SanaPipeline
from config import device 
import os

disease = "A high-resolution image of a diseased leaf showing visible symptoms of infection. The trifoliate leaf has three oval-shaped leaflets with noticeable yellowing (chlorosis) and irregular brown spots or lesions scattered across the surface. Some areas show signs of curling and wilting, and the leaf edges appear dry or necrotic. There may be signs of fungal growth or a powdery residue in certain areas. The overall green color is dull and patchy, indicating stress. The texture appears brittle and damaged. The leaf is isolated on a white background, lit with soft, natural lighting to highlight the details of the disease symptoms."
healthy = "A high-resolution image of a single leaf. The leaf is trifoliate, consisting of three oval-shaped leaflets with smooth edges and a slightly pointed tip. Each leaflet has a prominent central vein with smaller branching veins, giving it a delicate, netted appearance. The surface is slightly fuzzy with a fine layer of short hairs, and the leaf is a vibrant medium-green color. The texture is soft yet structured. The leaf is isolated on a white background with soft, natural lighting that accentuates its botanical details."

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to(device)
pipe.text_encoder.to(torch.bfloat16)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample.png")

def generate_image(image_health=True, image_path=file_path):
  if image_health:
    prompt = healthy
  else:
    prompt = disease
  image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=4.5,
        num_inference_steps=20,
        # generator=torch.Generator(device=device).manual_seed(42),
    )[0]

  image[0].save(image_path)
  return image[0]