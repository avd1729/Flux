import nltk
import logging

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    if not text or not text.strip():
        logger.warning("Empty text provided to chunker")
        return []
    
    logger.info(f"Chunking text of length {len(text)}")
    
    sentences = nltk.sent_tokenize(text)
    logger.info(f"Found {len(sentences)} sentences")
    
    out = []
    curr = ""
    
    for i, sentence in enumerate(sentences):
        if len(curr) + len(sentence) <= chunk_size:
            curr += " " + sentence if curr else sentence
        else:
            if curr.strip():
                out.append(curr.strip())
                logger.info(f"Created chunk {len(out)} of length {len(curr)}")
            
            overlap_start = max(0, i - 2)
            curr = " ".join(sentences[overlap_start:i+1])
    
    if curr.strip():
        out.append(curr.strip())
        logger.info(f"Created final chunk {len(out)} of length {len(curr)}")
    
    logger.info(f"Total chunks created: {len(out)}")
    return out