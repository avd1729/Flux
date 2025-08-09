from .embeddings import embed_texts
from .vector_store import load_index, search
import numpy as np
import logging

logger = logging.getLogger(__name__)

INDEX_DIM = 384

def deduplicate_results(results, similarity_threshold=0.9):
    if not results:
        return results
    
    deduped = []
    seen_texts = set()
    
    for score, meta in results:
        text = meta.get('text', '')
        
        text_words = set(text.lower().split())
        
        is_duplicate = False
        for seen_text in seen_texts:
            seen_words = set(seen_text.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(text_words.intersection(seen_words))
            union = len(text_words.union(seen_words))
            
            if union > 0:
                similarity = intersection / union
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            deduped.append((score, meta))
            seen_texts.add(text)
            logger.info(f"Added unique result: score={score:.4f}, source={meta.get('source', 'unknown')}")
        else:
            logger.info(f"Skipped duplicate result: score={score:.4f}")
    
    return deduped

def get_relevant_chunks(query, top_k=10):
    logger.info(f"Getting relevant chunks for query: {query[:50]}...")
    
    index = load_index(INDEX_DIM)
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    if index.ntotal == 0:
        logger.warning("Index is empty - no documents have been ingested yet")
        return []
    
    q_emb = embed_texts([query])[0]
    logger.info(f"Generated query embedding with shape: {q_emb.shape}")
    
    search_k = min(top_k * 3, index.ntotal)
    scores, metas = search(index, q_emb, top_k=search_k)
    results = list(zip(scores, metas))
    
    logger.info(f"Retrieved {len(results)} raw results")
    
    deduped_results = deduplicate_results(results, similarity_threshold=0.8)
    
    final_results = deduped_results[:top_k]
    
    logger.info(f"After deduplication: {len(final_results)} unique results")
    for i, (score, meta) in enumerate(final_results):
        logger.info(f"Result {i}: score={score:.4f}, source={meta.get('source', 'unknown')}")
    
    return final_results