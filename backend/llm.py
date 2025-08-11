# backend/llm.py
import logging
from transformers import RealmRetriever, RealmTokenizer, RealmReader, RealmReaderTokenizer
import torch

logger = logging.getLogger(__name__)

# Realm model names (Retriever + Reader)
REALM_RETRIEVER_MODEL = "google/realm-cc-news-pretrained"
REALM_READER_MODEL = "google/realm-cc-news-pretrained"

_retriever = None
_retriever_tokenizer = None
_reader = None
_reader_tokenizer = None


def get_realm_models():
    """Load Realm retriever and reader lazily."""
    global _retriever, _retriever_tokenizer, _reader, _reader_tokenizer
    if _retriever is None:
        logger.info("Loading Realm retriever and tokenizer...")
        _retriever = RealmRetriever.from_pretrained(REALM_RETRIEVER_MODEL)
        _retriever_tokenizer = RealmTokenizer.from_pretrained(REALM_RETRIEVER_MODEL)

    if _reader is None:
        logger.info("Loading Realm reader and tokenizer...")
        _reader = RealmReader.from_pretrained(REALM_READER_MODEL)
        _reader_tokenizer = RealmReaderTokenizer.from_pretrained(REALM_READER_MODEL)

    return _retriever, _retriever_tokenizer, _reader, _reader_tokenizer


def generate_answer(question, context_docs, max_tokens=128):
    """
    Use Realm to answer the question based on retrieved documents.
    context_docs: list of dicts with 'text' keys (retrieved chunks)
    """
    retriever, retr_tokenizer, reader, reader_tokenizer = get_realm_models()

    # Combine context docs into a single string
    combined_context = "\n\n".join(doc.get("text", "") for doc in context_docs)
    if not combined_context.strip():
        logger.warning("No context provided to Realm reader.")
        return "I don't know based on the provided materials."

    logger.info(f"Generating answer with Realm for question: {question}")

    # Tokenize question and context for the reader
    inputs = reader_tokenizer(
        [question],
        [combined_context],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = reader.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            num_beams=2,
            early_stopping=True,
        )

    answer = reader_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.info(f"Realm generated answer: {answer}")

    if not answer or answer.lower() in {"i don't know", "no", "yes"}:
        return "I don't know based on the provided materials."

    return answer
