import fitz
from chunker import split_text_into_chunks

def extract_pages(pdf_bytes):
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        yield page_idx + 1, text

def process_pdf_bytes(pdf_bytes, filename):
    chunks = []
    for page_num, page_text in extract_pages(pdf_bytes = pdf_bytes):
        for chunk_text in split_text_into_chunks(page_text, chunk_size=500, overlap=100):
            metadata = {
                "source": filename,
                "page": page_num
            }
            chunks.append(
                (chunk_text, metadata)
            )
            
    return chunks