# Retrieval-Augmented Generation (RAG) Project MVP

This is the MVP for a document-based question answering system. You can upload PDFs, process and chunk the content, generate embeddings using open-source models, store them in a vector store, and query to get relevant answers.

---

## Features

* Upload PDF documents
* Text extraction and chunking with overlap
* Generate vector embeddings with open-source models
* Store embeddings in a vector database (FAISS or similar)
* Query the vector store and retrieve relevant chunks
* Use open-source LLMs for answer generation

---

## Tech Stack

* Backend: FastAPI
* Frontend: Streamlit
* Vector Store: FAISS
* Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
* PDF Parsing: PyMuPDF
* Tokenization: NLTK (with punkt tokenizer)
* Language Model: Hugging Face Transformers

---

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/avd1729/college-assistant
   cd college-assistant
   ```

2. Create a virtual environment using `uv` (I assume you mean `python -m venv` or maybe `uvicorn`? If you meant something else, replace accordingly):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac  
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies from `pyproject.toml`:
   If you use Poetry:

   ```bash
   poetry install
   ```

   Or if you have `pip` only, you might need to convert that or use a requirements.txt instead.

4. Download required NLTK data (very important for tokenization):
   Run Python shell inside the virtual environment:

   ```python
   import nltk
   nltk.download('punkt')
   ```

5. Run the backend server:

   ```bash
   cd backend
   python main.py
   ```

6. Run the frontend app:
   In a new terminal window (with the venv activated):

   ```bash
   cd frontend
   python app.py
   ```

---

## Using the Makefile (Optional)

To automate the above steps, you can use the provided `Makefile` (for Linux/macOS or Windows with compatible shell):

| Command             | Description                |
| ------------------- | -------------------------- |
| `make venv`         | Create virtual environment |
| `make install`      | Install dependencies       |
| `make install-nltk` | Download NLTK tokenizer    |
| `make backend`      | Run backend server         |
| `make frontend`     | Run frontend app           |
| `make clean`        | Remove virtual environment |

---

**Example usage:**

```bash
make venv
source .venv/bin/activate
make install
make install-nltk
make backend
```

And in a new terminal window with the virtual environment activated:

```bash
make frontend
```

---

## Notes

* Ensure you activate the virtual environment before running the backend or frontend.
* The backend must be running before starting the frontend.
* For Windows users, consider using Git Bash or WSL to leverage the Makefile, or follow the manual steps above.

---


## Usage

* Upload PDFs through the frontend UI
* Backend processes and chunks documents, creates embeddings
* Ask questions via the frontend, which queries the backend
* Answers are generated using the vector store + local LLM

---

## Troubleshooting

* **NLTK punkt resource error:**
  Run `nltk.download('punkt')` as shown above.

* **Vector store missing or empty:**
  Make sure you upload PDFs and the backend completes processing before querying.

* **Environment issues:**
  Make sure your venv is activated and dependencies installed properly.


