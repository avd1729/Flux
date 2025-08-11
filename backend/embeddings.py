from transformers import RealmRetriever, RealmTokenizer
import torch
import numpy as np

MODEL_NAME = "google/realm-cc-news-pretrained"
_retriever = None
_tokenizer = None

def get_retriever():
    global _retriever, _tokenizer
    if _retriever is None:
        _retriever = RealmRetriever.from_pretrained(MODEL_NAME)
        _tokenizer = RealmTokenizer.from_pretrained(MODEL_NAME)
    return _retriever, _tokenizer

def embed_texts(texts):
    retriever, tokenizer = get_retriever()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = retriever.embed_passages(**inputs)
    return outputs.cpu().numpy().astype("float32")
