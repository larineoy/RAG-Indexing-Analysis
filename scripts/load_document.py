"""
Shared document loading functions for RAG indexing analysis.
Supports various document formats including DOCX, PDF, TXT, etc.
"""

import os
from typing import List, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Document
from llama_index.readers.file import DocxReader, PDFReader
from llama_index.core.readers import SimpleDirectoryReader

# LangChain imports
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader
)

# from config.settings import DOCUMENT_PATH, CORPUS_DIR
from config.settings import DOCUMENT_PATH, CORPUS_DIR

def load_document_llama_index(file_path: str) -> List[Document]:
    """
    Load document using LlamaIndex readers.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of LlamaIndex Document objects
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    file_extension = path_obj.suffix.lower()
    
    if file_extension == '.docx':
        reader = DocxReader()
        documents = reader.load_data(path_obj)
    elif file_extension == '.pdf':
        reader = PDFReader()
        documents = reader.load_data(path_obj)
    elif file_extension in ['.txt', '.md']:
        reader = SimpleDirectoryReader(input_files=[str(path_obj)])
        documents = reader.load_data()
    else:
        # Fallback to unstructured loader
        reader = SimpleDirectoryReader(input_files=[str(path_obj)])
        documents = reader.load_data()
    
    return documents


def load_document_langchain(file_path: str) -> List:
    """
    Load document using LangChain loaders.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of LangChain Document objects
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    file_extension = path_obj.suffix.lower()
    
    if file_extension == '.docx':
        loader = Docx2txtLoader(str(path_obj))
    elif file_extension == '.pdf':
        loader = PyPDFLoader(str(path_obj))
    elif file_extension in ['.txt', '.md']:
        loader = TextLoader(str(path_obj))
    else:
        # Fallback to unstructured loader
        loader = UnstructuredFileLoader(str(path_obj))
    
    documents = loader.load()
    return documents


def get_document_text(documents: List) -> str:
    """
    Extract text content from documents.
    
    Args:
        documents: List of document objects (either LlamaIndex or LangChain)
        
    Returns:
        Combined text content
    """
    if not documents:
        return ""
    
    # Check if it's LlamaIndex documents
    if hasattr(documents[0], 'text'):
        return "\n\n".join([doc.text for doc in documents])
    
    # Check if it's LangChain documents
    if hasattr(documents[0], 'page_content'):
        return "\n\n".join([doc.page_content for doc in documents])
    
    return ""


def load_corpus_llama_index(corpus_dir: str = CORPUS_DIR) -> List[Document]:
    """
    Load all documents from a corpus directory using LlamaIndex.
    
    Args:
        corpus_dir: Directory containing documents
        
    Returns:
        List of LlamaIndex Document objects
    """
    reader = SimpleDirectoryReader(input_dir=corpus_dir)
    documents = reader.load_data()
    return documents


def load_corpus_langchain(corpus_dir: str = CORPUS_DIR) -> List:
    """
    Load all documents from a corpus directory using LangChain.
    
    Args:
        corpus_dir: Directory containing documents
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        return documents
    
    for file_path in corpus_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.docx', '.pdf', '.txt', '.md']:
            try:
                doc_list = load_document_langchain(str(file_path))
                documents.extend(doc_list)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents 