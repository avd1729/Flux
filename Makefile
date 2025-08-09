# Makefile for RAG Project MVP

.PHONY: venv install-nltk backend frontend clean help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Makefile commands:"
	@echo "  venv         - Create and activate virtual environment"
	@echo "  install      - Install dependencies from pyproject.toml"
	@echo "  install-nltk - Download nltk punkt tokenizer"
	@echo "  backend      - Run backend server"
	@echo "  frontend     - Run frontend app"
	@echo "  clean        - Remove virtual environment"

venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created. Activate it with: source $(VENV)/bin/activate"

install: venv
	$(PIP) install --upgrade pip
	poetry install

install-nltk:
	$(PYTHON) -c "import nltk; nltk.download('punkt')"

backend:
	cd backend && $(PYTHON) main.py

frontend:
	cd frontend && $(PYTHON) app.py

clean:
	rm -rf $(VENV)
