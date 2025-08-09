# tests/test_model.py
import pytest
from backend.embeddings import model

def test_model_returns_sentence_transformer():
    # Get the model instance
    m = model()

    # Check type without importing SentenceTransformer directly
    assert hasattr(m, "encode"), "Model should have an 'encode' method"

def test_model_encoding_sample():
    m = model()
    sentence = ["This is a test sentence."]
    embedding = m.encode(sentence)

    assert embedding is not None, "Embedding should not be None"
    assert len(embedding) == 1, "Embedding should return one vector"
    assert embedding[0].ndim == 1, "Each embedding should be 1D"
    assert embedding[0].size > 0, "Embedding vector should not be empty"
