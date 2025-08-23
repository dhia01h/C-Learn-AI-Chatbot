import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "db"
DOCS_DIR = "docs"

def extract_text_from_pdfs(folder_path=DOCS_DIR):
    full_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf = fitz.open(pdf_path)
            for page in pdf:
                full_text += page.get_text()
    return full_text

def split_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def build_vectorstore():
    text = extract_text_from_pdfs()
    chunks = split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=PERSIST_DIR)
    vectordb.persist()
    return vectordb

def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)



