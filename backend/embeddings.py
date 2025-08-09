from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts):
    m = model()
    embs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype("float32") 