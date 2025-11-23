"""
LlamaIndex TreeIndex example for RAG indexing analysis.
Uses Ollama LLM and HuggingFace embeddings.
"""

from llama_index.core import TreeIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from scripts.load_document import load_document_llama_index
from config.settings import DOCUMENT_PATH, LLM_MODEL, EMBED_MODEL


def main():
    # 1. Load your data as a list of Document objects
    print(f"Loading document: {DOCUMENT_PATH}")
    documents = load_document_llama_index(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")

    # 2. Set up LLM and embedding model
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)  # 5 minutes
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # 3. Build the TreeIndex
    print("Building TreeIndex (this may take a while for large documents)...")
    index = TreeIndex.from_documents(documents, show_progress=True)
    print("TreeIndex built!")

    # 4. Create a query engine (default is tree summarization)
    query_engine = index.as_query_engine()

    # 5. Example query
    query = "What are the main specifications of the 5.56mm ammunition?"
    print(f"\nQuery: {query}")
    response = query_engine.query(query)
    print(f"\nResponse:\n{response}")

    # 6. (Optional) Save the index for later use
    # index.storage_context.persist(persist_dir="outputs/llama_tree_index")

def get_query_engine():
    documents = load_document_llama_index(DOCUMENT_PATH)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    index = TreeIndex.from_documents(documents, show_progress=False)
    query_engine = index.as_query_engine()
    return query_engine

if __name__ == "__main__":
    main() 