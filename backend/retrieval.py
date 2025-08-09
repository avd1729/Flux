from .embeddings import embed_texts
from .vector_store import load_index, search
import numpy as np
import logging

logger = logging.getLogger(__name__)

INDEX_DIM = 384

def get_relevant_chunks(query, top_k=10):
    logger.info(f"Getting relevant chunks for query: {query[:50]}...")
    
    # Load index fresh each time to get latest data
    index = load_index(INDEX_DIM)
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    if index.ntotal == 0:
        logger.warning("Index is empty - no documents have been ingested yet")
        return []
    
    q_emb = embed_texts([query])[0]
    logger.info(f"Generated query embedding with shape: {q_emb.shape}")
    
    scores, metas = search(index, q_emb, top_k=top_k)
    results = list(zip(scores, metas))
    
    logger.info(f"Retrieved {len(results)} results")
    for i, (score, meta) in enumerate(results[:3]):  # Log first 3
        logger.info(f"Result {i}: score={score:.4f}, source={meta.get('source', 'unknown')}")
    
    return results