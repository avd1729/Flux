import backend.retrieval

def test_get_relevant_chunks(monkeypatch):
    # Mock embed_texts to return a fixed embedding
    def mock_embed_texts(texts):
        assert texts == ["test query"]  # Ensure query is passed correctly
        return [[0.1, 0.2, 0.3]]

    # Mock search to return predictable results
    def mock_search(index, q_emb, top_k):
        assert q_emb == [0.1, 0.2, 0.3]
        assert top_k == 3
        return [0.9, 0.8, 0.7], [
            {"page": 1, "text": "Chunk 1"},
            {"page": 2, "text": "Chunk 2"},
            {"page": 3, "text": "Chunk 3"},
        ]

    # Patch dependencies
    monkeypatch.setattr(backend.retrieval, "embed_texts", mock_embed_texts)
    monkeypatch.setattr(backend.retrieval, "search", mock_search)

    # Call function
    result = backend.retrieval.get_relevant_chunks("test query")

    # Verify
    expected = [
        (0.9, {"page": 1, "text": "Chunk 1"}),
        (0.8, {"page": 2, "text": "Chunk 2"}),
        (0.7, {"page": 3, "text": "Chunk 3"}),
    ]
    assert result == expected
