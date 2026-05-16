from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI

