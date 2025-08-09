import fitz
from .chunker import split_text_into_chunks
import logging

logger = logging.getLogger(__name__)

def extract_pages(pdf_bytes):
    logger.info(f"Opening PDF with {len(pdf_bytes)} bytes")
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"PDF opened successfully, {len(doc)} pages")
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        return

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        logger.info(f"Page {page_idx + 1}: extracted {len(text)} characters")
        if text.strip():
            logger.info(f"Page {page_idx + 1} preview: {text[:100]}...")
        else:
            logger.warning(f"Page {page_idx + 1} is empty!")
        yield page_idx + 1, text

def process_pdf_bytes(pdf_bytes, filename):
    logger.info(f"Processing PDF: {filename}")
    chunks = []
    
    try:
        for page_num, page_text in extract_pages(pdf_bytes=pdf_bytes):
            if not page_text.strip():
                logger.warning(f"Page {page_num} has no text content")
                continue
                
            page_chunks = split_text_into_chunks(page_text, chunk_size=500, overlap=100)
            logger.info(f"Page {page_num}: created {len(page_chunks)} chunks")
            
            for i, chunk_text in enumerate(page_chunks):
                if not chunk_text.strip():
                    logger.warning(f"Empty chunk {i} on page {page_num}")
                    continue
                    
                metadata = {
                    "source": filename,
                    "page": page_num,
                    "text": chunk_text
                }
                chunks.append((chunk_text, metadata))
                
                if len(chunks) <= 3:  # Log first few chunks
                    logger.info(f"Chunk {len(chunks)}: {len(chunk_text)} chars, preview: {chunk_text[:50]}...")
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks