import faiss
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.idx")
META_PATH = os.path.join(INDEX_DIR, "metadata.jsonl")

def init(dim):
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        logger.info(f"Created index directory: {INDEX_DIR}")
    index = faiss.IndexFlatIP(dim)
    logger.info(f"Initialized new FAISS index with dimension {dim}")
    return index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"Saved index to {INDEX_PATH}")

def load_index(dim):
    if os.path.exists(INDEX_PATH):
        logger.info(f"Loading existing index from {INDEX_PATH}")
        return faiss.read_index(INDEX_PATH)
    else:
        logger.info("No existing index found, creating new one")
        return init(dim)

def add(index, vectors, metadatas):
    logger.info(f"Adding {len(vectors)} vectors to index")
    
    # vectors: numpy array (n,d) float32; normalize for cosine
    vectors_normalized = vectors.copy()
    faiss.normalize_L2(vectors_normalized)
    index.add(vectors_normalized)
    
    logger.info(f"Index now contains {index.ntotal} vectors")

    # Append metadata to file
    with open(META_PATH, "a", encoding='utf-8') as f:
        for i, m in enumerate(metadatas):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            if i == 0: 
                logger.info(f"First metadata entry: {m}")

def search(index, qvec, top_k=5):
    if index.ntotal == 0:
        logger.warning("Index is empty, no vectors to search")
        return [], []
    
    q = qvec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    logger.info(f"Searching index with {index.ntotal} vectors for top {top_k} results")
    D, I = index.search(q, min(top_k, index.ntotal))
    
    metas = []
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding='utf-8') as f:
            lines = f.readlines()
            logger.info(f"Loaded {len(lines)} metadata lines")
        
        for idx in I[0]:
            if 0 <= idx < len(lines):
                try:
                    meta = json.loads(lines[idx])
                    metas.append(meta)
                    logger.info(f"Retrieved metadata for index {idx}: keys={list(meta.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata line {idx}: {e}")
                    metas.append({})
            else:
                logger.warning(f"Index {idx} is out of range for metadata (have {len(lines)} lines)")
                metas.append({})
    else:
        logger.warning(f"Metadata file {META_PATH} does not exist")
        metas = [{}] * len(I[0])
    
    scores = D[0].tolist()
    logger.info(f"Search completed, found {len(metas)} results with scores: {scores}")
    
    return scores, metas