from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import shutil
import lancedb
import pandas as pd
from utils import process_and_embed, save_to_lancedb, answer_query

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


db = lancedb.connect(
    uri="db://rag-8iqoag",
    api_key=os.getenv("LANCEDB_API_KEY"),
    region="us-east-1",
)


TABLE_NAME = "documents2"


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedded_chunks = process_and_embed(file_path)

    save_to_lancedb(embedded_chunks, db=db)

    return {
        "filename": file.filename,
        "total_chunks": len(embedded_chunks),
        "first_chunk_preview": embedded_chunks[0] if embedded_chunks else {},
    }


@app.post("/query-by-user")
async def query_question(request: Question):
    table = db.open_table(TABLE_NAME)
    answer = answer_query(request.question, table=table)
    return {"answer": answer}
