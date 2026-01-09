from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List

def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Loads a PDF file and returns a list of Document objects.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []
