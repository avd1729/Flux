import backend.retrieval
import numpy as np

class FakeIndex:
    def __init__(self, ntotal):
        self.ntotal = ntotal

def test_get_relevant_chunks(monkeypatch):
    # Mock Realm query embedding
    def mock_embed_texts(texts):
        assert texts == ["test query"]
        return [np.array([0.1, 0.2, 0.3])]  # shape: (3,) for simplicity in test

    # Mock vector search
    def mock_search(index, q_emb, top_k):
        assert list(q_emb) == [0.1, 0.2, 0.3]
        assert top_k == 10
        return [0.9, 0.8, 0.7], [
            {"page": 1, "text": "Chunk 1"},
            {"page": 2, "text": "Chunk 2"},
            {"page": 3, "text": "Chunk 3"},
        ]

    # Mock index loader
    def mock_load_index(dim):
        assert dim == 768  # Realm embedding dimension
        return FakeIndex(ntotal=10)

    # Apply monkeypatches
    monkeypatch.setattr(backend.retrieval, "embed_texts", mock_embed_texts)
    monkeypatch.setattr(backend.retrieval, "search", mock_search)
    monkeypatch.setattr(backend.retrieval, "load_index", mock_load_index)

    result = backend.retrieval.get_relevant_chunks("test query")

    expected = [
        (0.9, {"page": 1, "text": "Chunk 1"}),
        (0.8, {"page": 2, "text": "Chunk 2"}),
        (0.7, {"page": 3, "text": "Chunk 3"}),
    ]
    assert result == expected
