import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def create_vector_store(documents: list[Document]):
    """
    Splits documents into chunks, generates embeddings, and stores them in a FAISS vector store.
    Returns the vector store.
    """
    # 1. Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return None

    # 2. Embeddings
    # Using a local sentence-transformer model that is free and runs on CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store
