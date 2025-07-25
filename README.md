# 📄 RAG-Enhanced Document & Image QA System

This project is a **Retrieval-Augmented Generation (RAG)** based intelligent system that allows users to upload a document or image file and ask natural language questions about its content. It leverages embedding-based semantic search using **LanceDB** and provides accurate, context-aware answers using **LLMs**.

---

## 🚀 Features

- 📤 Upload support for **text documents** and **image files**
- 🧾 **OCR** (Optical Character Recognition) to extract text from images
- ✂️ **Text chunking** and **embedding generation**
- 💾 Vector storage with **LanceDB**
- 🔍 **Semantic search** for context retrieval
- 🤖 Final answer generation using **RAG (Retrieval-Augmented Generation)** with an LLM

---

## 🛠️ How It Works

1. **User Uploads** a document (PDF, TXT) or an image (JPG, PNG).
2. If an image is uploaded, **OCR** is applied to extract text using tools like Tesseract or easyOCR.
3. The extracted text is **chunked** and **converted to embeddings** using an embedding model (e.g., Mistral or any sentence-transformer).
4. These embeddings are **stored in LanceDB** as vectors.
5. When the user asks a **question**, it is embedded and **matched against the stored vectors**.
6. The most similar chunks are retrieved and passed to the **LLM**, which **generates an answer** using the retrieved context.

---

## 📦 Tech Stack

- **Python**
- **LanceDB** – for storing and retrieving document embeddings
- **Tesseract / easyOCR** – for OCR on image files
- **Sentence Transformers / Mistral / HuggingFace** – for embedding generation
- **LangChain / Custom RAG Logic** – for Retrieval-Augmented Generation
- **FastAPI** – for backend API

---

## 🖼️ Example Workflow

1. **Upload**: `invoice.png`  
2. **OCR Text**: _"Total amount: ₹2,450. Due by: 15th Aug"_  
3. **Chunking + Embedding → LanceDB**
4. **User Query**: _"What is the due date on this invoice?"_
5. **RAG Output**: _"The due date on this invoice is 15th Aug."_

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG-DocQA.git
cd RAG-DocQA
```


### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Run the backend (FastAPI or your chosen framework)

```bash
uvicorn app.main:app --reload
```


