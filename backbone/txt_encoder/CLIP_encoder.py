import torch
from transformers import CLIPModel, CLIPProcessor
import os


def encode_text_single(text, model, processor, device):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        outputs = model.text_model(**inputs)
        text_embeddings = outputs.last_hidden_state

    return text_embeddings
