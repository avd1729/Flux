import faiss
import numpy as np
import json
import os

INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.idx")
META_PATH = os.path.join(INDEX_DIR, "metadata.jsonl")

def init(dim):
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    index = faiss.IndexFlatIP(dim)
    return index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_index(dim):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return init(dim)

def add(index, vectors, metadatas):
    # vectors: numpy array (n,d) float32; normalize for cosine
    faiss.normalize_L2(vectors)
    index.add(vectors)

    with open(META_PATH, "a") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def search(index, qvec, top_k=5):
    q = qvec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    D, I = index.search(q, top_k)
    metas = []

    with open(META_PATH) as f:
        lines = f.readlines()
    for idx in I[0]:
        if idx < len(lines):
            metas.append(json.loads(lines[idx]))
    return D[0].tolist(), metas
