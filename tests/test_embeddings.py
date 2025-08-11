import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

@patch("backend.embeddings.get_retriever")
def test_embedding_mocked(mock_get_retriever):
    # Create mock retriever and tokenizer
    mock_retriever = MagicMock()
    mock_tokenizer = MagicMock()

    # Fake tokenizer output (what tokenizer(...) normally returns)
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Fake retriever output (what embed_passages normally returns)
    mock_retriever.embed_passages.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    # Make get_retriever() return our tuple
    mock_get_retriever.return_value = (mock_retriever, mock_tokenizer)

    from backend.embeddings import embed_texts
    emb = embed_texts(["hello"])

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 3)
    assert np.allclose(emb[0], [0.1, 0.2, 0.3])
