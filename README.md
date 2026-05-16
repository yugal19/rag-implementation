# Hallucination Detection in RAG-Based Actuarial Agents

## Overview

This project is a Retrieval-Augmented Generation (RAG)-based actuarial assistant designed for intelligent insurance policy analysis with integrated hallucination detection and verification. The system processes insurance-related documents, extracts and stores semantic embeddings, retrieves contextually relevant policy clauses, and generates grounded responses using Large Language Models (LLMs).

To improve factual reliability, the architecture incorporates a dual-stage hallucination detection and cross-model verification framework. Generated responses are audited for fabrication, contradiction, incorrect inference, missing inference, and partial grounding before being independently verified using a separate LLM. The system is built using FastAPI, LangChain, LanceDB, Mistral AI, and Groq-hosted Llama 3.1 models, providing a scalable and reliable pipeline for AI-driven actuarial and insurance document analysis.

Built using:

- FastAPI
- LangChain
- LanceDB
- Mistral AI
- Groq Llama 3.1

---

# Features

- Multi-format document ingestion (PDF, DOCX, PPTX, Images)
- Semantic chunking and embeddings
- LanceDB vector storage
- Context-aware RAG pipeline
- Hallucination detection
- Cross-model verification
- FastAPI backend APIs

---

# Workflow

```text
Document Upload
    ↓
Text Extraction
    ↓
Chunking & Embeddings
    ↓
LanceDB Storage
    ↓
Semantic Retrieval
    ↓
LLM Response Generation
    ↓
Hallucination Detection
    ↓
Cross-Model Verification
    ↓
Final Verified Response
```

---

# Installation

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create `.env`

```env
MISTRAL_API_KEY=your_key
GROQ_API_KEY=your_key
LANCEDB_API_KEY=your_key
HF_TOKEN=your_key
```

---

# Run Server

```bash
uvicorn main:app --reload
```

---

# API Endpoints

## Upload Document

```http
POST /upload
```

## Query System

```http
POST /query-by-user
```

Example Request:

```json
{
  "question": "What is the waiting period for pre-existing conditions?"
}
```

---

# Tech Stack

- Python
- FastAPI
- LangChain
- LanceDB
- Mistral AI
- Groq API
- Llama 3.1

---

# Project Structure

```text
project/
│
├── main.py
├── utils.py
├── uploads/
├── requirements.txt
└── README.md
```

---

# Conclusion

The project introduces a hallucination-aware RAG architecture for insurance policy analysis by combining semantic retrieval, constrained generation, hallucination auditing, and independent verification to improve reliability in AI-generated responses.
