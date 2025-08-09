from embeddings import embed_texts
from vector_store import load_index, search
import numpy as np

INDEX_DIM = 384

index = load_index(INDEX_DIM)

def get_relevant_chunks(query, top_k=3):
    q_emb = embed_texts([query])[0]
    scores, metas = search(index, q_emb, top_k=top_k)
    return list(zip(scores, metas))