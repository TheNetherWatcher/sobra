from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from config import device

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

def vlm(prompt, img_path=None):
    
    if img_path:
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "image", "path": img_path},
                  {"type": "text", "text": prompt},
              ]
          },
      ]
    else:
      messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "path": img_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    # print(inputs)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    description = generated_texts[0]
    description = description.split("Assistant:")[1].strip()
    return description