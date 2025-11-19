import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


def encode_text_single_with_bert(text, model, tokenizer, device):
    inputs = tokenizer(
        text=[text],
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        text_embeddings = outputs.last_hidden_state

    return text_embeddings
