import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from req import *
from dotenv import load_dotenv

load_dotenv()

import re
import json

from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import lancedb


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


def save_to_lancedb(embedded_chunks, db, table_name="documents3"):
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
        return {
            "answer": "Not enough information.",
            "hallucination_analysis": {
                "hallucination": False,
                "error_type": "no_context",
                "confidence_score": 1.0,
                "reason": "No relevant context retrieved",
            },
            "retrieved_chunks": [],
            "conditions": {"retrieval_failure": True, "context_length": 0},
        }

    context = " ".join([r["text"].replace("\n", " ") for r in results])
    clean_chunks = [r["text"].replace("\n", " ").strip() for r in results]

    prompt = f"""
You are an expert insurance policy analyst.

STRICT RULES:
1. Use ONLY the given policy context.
2. If ANY relevant clause exists → you MUST answer using it.
3. DO NOT say "Not enough information" if partial evidence exists.
4. Always extract the most likely interpretation from policy wording.
5. Mention exceptions if present.
6. If absolutely NOTHING relevant exists → say "Not enough information".

7. DO NOT use external knowledge.

OUTPUT FORMAT:
Answer: <final answer>
Reasoning: <brief reasoning grounded in context>

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    try:
        answer_part = raw_output.split("Answer:")[1].split("Reasoning:")[0].strip()
    except:
        answer_part = raw_output

    hallucination_result = detect_hallucination(context, query, raw_output)

    verification = llm_verify(query, raw_output, context)

    conditions = {
        "retrieval_failure": len(results) == 0,
        "low_retrieval": len(results) < 2,
        "context_length": len(context),
        "ambiguous_query": len(query.split()) < 4,
    }

    return {
        "answer": answer_part,
        "hallucination_analysis": hallucination_result,
        "verification": verification,
        "retrieved_chunks": clean_chunks,
        "conditions": conditions,
    }


def detect_hallucination(context: str, query: str, answer: str):
    context = context.replace("\n", " ")
    answer = answer.replace("\n", " ")

    prompt = f"""
You are a strict auditor for RAG-based insurance systems.

Your job is to detect hallucinations AND reasoning failures.
You must evaluate NOT just correctness, but whether the answer uses the MOST RELEVANT policy clause.

---
Context:
{context}

Question:
{query}

Answer:
{answer}
---
Follow these steps STRICTLY:

STEP 1: Identify what the question is asking (e.g., waiting period, coverage, limits).

STEP 2: Identify ALL relevant clauses in the context related to the question.

STEP 3: Check whether the answer:
    a) Uses the correct clause
    b) Ignores a more relevant clause
    c) Mixes unrelated clauses

---

Evaluation Rules:

1. Fabrication:
   - Answer includes information NOT present in context

2. Contradiction:
   - Answer conflicts with context

3. Incorrect Inference (IMPORTANT):
   - Answer uses context BUT applies the WRONG clause
   - OR ignores the most relevant clause
   - OR uses unrelated information (e.g., wrong waiting period)

4. Missing Inference:
   - Context contains enough information, but model failed to derive answer
   - OR incorrectly says "Not enough information"

5. Partial Grounding:
   - Answer is partially correct but misses key constraints or conditions

6. Correct:
   - Answer is fully supported AND uses the most relevant clause correctly

---

Special Rule (VERY IMPORTANT):
If multiple clauses exist, the answer MUST rely on the MOST RELEVANT one.
If it relies on a less relevant or unrelated clause → hallucination = true

---

Return EXACTLY this JSON:

{{
  "hallucination": true or false,
  "error_type": "fabrication | contradiction | incorrect_inference | missing_inference | partial_grounding | none",
  "confidence_score": number between 0 and 1,
  "reason": "short explanation mentioning whether correct clause was used"
}}
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    try:
        clean_content = re.sub(
            r"^```json\s*|```$", "", content, flags=re.MULTILINE
        ).strip()

        result = json.loads(clean_content)

        if "hallucination" not in result:
            raise ValueError("Invalid JSON structure")

    except Exception as e:
        result = {
            "hallucination": True,
            "error_type": "parsing_error",
            "confidence_score": 0.0,
            "reason": f"Parsing failed: {str(e)}",
        }

    return result


def llm_verify(query: str, answer: str, context: str):
    context = context.replace("\n", " ")
    answer = answer.replace("\n", " ")

    prompt = f"""
You are an independent verification system.

Your job is to DOUBLE-CHECK whether the answer is correct using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
{answer}

---

Verification Rules:

1. If answer is fully supported → correct = true
2. If answer uses wrong clause → correct = false
3. If answer says "Not enough information" but context has answer → correct = false
4. If answer adds extra info → correct = false

---

Return JSON ONLY:

{{
  "correct": true or false,
  "confidence": number between 0 and 1,
  "reason": "short explanation"
}}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        clean_content = re.sub(r"^```json\s*|```$", "", content).strip()
        result = json.loads(clean_content)

        if "correct" not in result:
            raise ValueError("Invalid JSON")

    except Exception as e:
        result = {
            "correct": False,
            "confidence": 0.0,
            "reason": f"Parsing failed: {str(e)}",
        }

    return result
