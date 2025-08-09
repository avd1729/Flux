import os
import sys
from unittest.mock import patch
from backend.chunker import split_text_into_chunks

# # Add the project root to sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_split_text_into_chunks_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_text_into_chunks(text, chunk_size=40, overlap=10)
    
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) >= 1
    assert all(len(c.strip()) > 0 for c in chunks)


def test_split_text_into_chunks_small_text():
    text = "Short sentence."
    chunks = split_text_into_chunks(text, chunk_size=500, overlap=100)
    
    assert chunks == [text]


@patch("nltk.sent_tokenize", return_value=["Sentence one.", "Sentence two.", "Sentence three."])
def test_split_text_into_chunks_overlap(mock_tokenize):
    text = "dummy"
    chunks = split_text_into_chunks(text, chunk_size=20, overlap=5)
    
    assert len(chunks) >= 2
    assert all("Sentence" in chunk for chunk in chunks)

