from fastapi import FastAPI, UploadFile
from .ingestion import process_pdf_bytes
from .vector_store import load_index, add, save_index
from .embeddings import embed_texts
from .retrieval import get_relevant_chunks
from .llm import generate_answer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    logger.info(f"Received file: {file.filename}, size: {len(content)} bytes")
    
    # Do ingestion synchronously for debugging
    try:
        result = handle_ingest(content, file.filename)
        return {"status": "completed", "message": "File processed successfully"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def handle_ingest(file_bytes, filename):
    try:
        logger.info(f"Starting ingestion for {filename}")
        chunks = process_pdf_bytes(file_bytes, filename)
        logger.info(f"Generated {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks generated from PDF!")
            return
        
        if chunks:
            logger.info(f"First chunk preview: {chunks[0][0][:100]}...")
            logger.info(f"First chunk metadata: {chunks[0][1]}")
        
        texts = [c for c, _ in chunks]
        metas = [m for _, m in chunks]
        
        logger.info(f"Extracted {len(texts)} texts and {len(metas)} metadata entries")
        
        if not texts:
            logger.error("No texts extracted from chunks!")
            return
        
        embs = embed_texts(texts)
        logger.info(f"Generated embeddings shape: {embs.shape}")
        
        index = load_index(384)
        logger.info(f"Index before adding: {index.ntotal} vectors")
        
        add(index, embs, metas)
        logger.info(f"Index after adding: {index.ntotal} vectors")
        
        save_index(index)
        logger.info("Ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

@app.get("/ask")
def ask(q: str, top_k: int = 3):
    logger.info(f"Question: {q}")
    
    hits = get_relevant_chunks(q, top_k=top_k)
    logger.info(f"Retrieved {len(hits)} hits")
    
    for i, (score, meta) in enumerate(hits):
        logger.info(f"Hit {i}: score={score}, meta keys={meta.keys()}")
        if 'text' in meta:
            logger.info(f"Hit {i} text preview: {meta['text'][:100]}...")
        else:
            logger.info(f"Hit {i} has no 'text' key! Available keys: {list(meta.keys())}")
    
    # Build context
    context = "\n\n".join([
        f"{h.get('text', '[NO TEXT]')} (Source: {h.get('source', 'unknown')}, page {h.get('page', 'N/A')})"
        for _, h in hits
    ])
    
    logger.info(f"Context length: {len(context)}")
    logger.info(f"Context preview: {context[:200]}...")
    
    answer = generate_answer(q, context)
    logger.info(f"Generated answer: {answer}")
    
    return {"answer": answer, "sources": [h for _, h in hits], "debug": {"context_preview": context[:200]}}