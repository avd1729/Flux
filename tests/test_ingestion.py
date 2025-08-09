import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from backend.ingestion import extract_pages, process_pdf_bytes

# # Add the project root to sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_extract_pages_yields_text():
    # Mock fitz document and pages
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page text"

    mock_doc = [mock_page, mock_page]  # two pages

    with patch("fitz.open", return_value=mock_doc):
        pdf_bytes = b"%PDF-1.4 dummy data"
        pages = list(extract_pages(pdf_bytes))
        
        assert len(pages) == 2
        assert pages[0] == (1, "Page text")
        assert pages[1] == (2, "Page text")


@patch("backend.ingestion.extract_pages", return_value=[(1, "First page text."), (2, "Second page text.")])
@patch("backend.ingestion.split_text_into_chunks", side_effect=lambda text, **kwargs: [text.upper()])
def test_process_pdf_bytes_calls_split_and_returns_chunks(mock_split, mock_extract):
    pdf_bytes = b"dummy"
    filename = "test.pdf"

    from backend.ingestion import process_pdf_bytes

    chunks = process_pdf_bytes(pdf_bytes, filename)

    assert len(chunks) == 2
    for chunk_text, metadata in chunks:
        assert chunk_text.isupper()
        assert metadata["source"] == filename
        assert "page" in metadata

    mock_extract.assert_called_once_with(pdf_bytes=pdf_bytes)
    assert mock_split.call_count == 2


