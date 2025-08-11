import faiss
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

BASE_INDEX_DIR = "data/index"

def _get_paths(realm_id):
    """Return index and metadata paths for a specific realm."""
    index_dir = os.path.join(BASE_INDEX_DIR, realm_id)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        logger.info(f"Created index directory for realm '{realm_id}': {index_dir}")
    index_path = os.path.join(index_dir, "faiss.idx")
    meta_path = os.path.join(index_dir, "metadata.jsonl")
    return index_path, meta_path


def init(dim):
    index = faiss.IndexFlatIP(dim)
    logger.info(f"Initialized new FAISS index with dimension {dim}")
    return index


def save_index(index, index_path):
    faiss.write_index(index, index_path)
    logger.info(f"Saved index to {index_path}")


def load_index(realm_id, dim):
    index_path, _ = _get_paths(realm_id)
    if os.path.exists(index_path):
        logger.info(f"Loading existing index for realm '{realm_id}' from {index_path}")
        return faiss.read_index(index_path)
    else:
        logger.info(f"No existing index found for realm '{realm_id}', creating new one")
        return init(dim)


def add(realm_id, index, vectors, metadatas):
    index_path, meta_path = _get_paths(realm_id)

    logger.info(f"Adding {len(vectors)} vectors to index for realm '{realm_id}'")
    vectors_normalized = vectors.copy().astype("float32")
    faiss.normalize_L2(vectors_normalized)
    index.add(vectors_normalized)

    logger.info(f"Index for realm '{realm_id}' now contains {index.ntotal} vectors")

    # Save index after adding
    save_index(index, index_path)

    # Append metadata
    with open(meta_path, "a", encoding="utf-8") as f:
        for i, m in enumerate(metadatas):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            if i == 0:
                logger.info(f"First metadata entry for realm '{realm_id}': {m}")


def search(realm_id, index, qvec, top_k=5):
    _, meta_path = _get_paths(realm_id)

    if index.ntotal == 0:
        logger.warning(f"Index for realm '{realm_id}' is empty, no vectors to search")
        return [], []

    q = qvec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    logger.info(f"Searching index for realm '{realm_id}' with {index.ntotal} vectors for top {top_k} results")
    D, I = index.search(q, min(top_k, index.ntotal))

    metas = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            lines = f.readlines()
            logger.info(f"Loaded {len(lines)} metadata lines for realm '{realm_id}'")

        for idx in I[0]:
            if 0 <= idx < len(lines):
                try:
                    meta = json.loads(lines[idx])
                    metas.append(meta)
                    logger.info(f"Retrieved metadata for index {idx} in realm '{realm_id}': keys={list(meta.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata line {idx} for realm '{realm_id}': {e}")
                    metas.append({})
            else:
                logger.warning(f"Index {idx} is out of range for metadata in realm '{realm_id}'")
                metas.append({})
    else:
        logger.warning(f"Metadata file {meta_path} does not exist for realm '{realm_id}'")
        metas = [{}] * len(I[0])

    scores = D[0].tolist()
    logger.info(f"Search completed in realm '{realm_id}', found {len(metas)} results with scores: {scores}")

    return scores, metas
