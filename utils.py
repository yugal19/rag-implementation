import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from dotenv import load_dotenv


import lancedb

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
hf_token = os.getenv("HF_TOKEN")
embedding_model = MistralAIEmbeddings(api_key=mistral_api_key)
llm = ChatMistralAI(api_key=mistral_api_key)


def extract_text_from_file(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        loader = UnstructuredImageLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def process_and_embed(file_path: str):
    documents = extract_text_from_file(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = MistralAIEmbeddings(api_key=mistral_api_key)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    embedded_chunks = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vectors[i],
        }
        for i, chunk in enumerate(chunks)
    ]

    return embedded_chunks


def save_to_lancedb(embedded_chunks, db, table_name="documents2"):
    records = []
    for chunk in embedded_chunks:
        metadata = chunk["metadata"]
        if not isinstance(metadata, dict):
            metadata = {"source": str(metadata)}
        records.append(
            {"text": chunk["text"], "metadata": metadata, "vector": chunk["embedding"]}
        )

    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(records)
    else:
        table = db.create_table(table_name, data=records)

    return table


def answer_query(query: str, table, top_k: int = 5):
    query_vector = embedding_model.embed_query(query)
    results = table.search(query_vector).limit(top_k).to_list()
    if not results:
        return "No relevant context found."

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content.replace("\n", " ").strip()
