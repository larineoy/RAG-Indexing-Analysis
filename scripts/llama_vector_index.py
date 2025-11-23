"""
LlamaIndex VectorStoreIndex example for RAG indexing analysis.
Uses Ollama LLM and HuggingFace embeddings.
"""

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings import HuggingFaceEmbedding
# from llama_index.embeddings.langchain import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from scripts.load_document import load_document_llama_index
# from scripts.load_document import load_document_llama_index
from config.settings import DOCUMENT_PATH, LLM_MODEL, EMBED_MODEL, TOP_K


def main():
    # 1. Load your data as a list of Document objects
    print(f"Loading document: {DOCUMENT_PATH}")
    documents = load_document_llama_index(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")

    # 2. Set up LLM and embedding model
    Settings.llm = Ollama(model=LLM_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # 3. Build the VectorStoreIndex
    print("Building VectorStoreIndex (this may take a while for large documents)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("Index built!")

    # 4. Create a query engine with top_k retrieval
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    # 5. Example query
    query = "What are the main specifications of the 5.56mm ammunition?"
    print(f"\nQuery: {query}")
    response = query_engine.query(query)
    print(f"\nResponse:\n{response}")

    # 6. (Optional) Save the index for later use
    # index.storage_context.persist(persist_dir="outputs/llama_vector_index")

def get_query_engine():
    documents = load_document_llama_index(DOCUMENT_PATH)
    Settings.llm = Ollama(model=LLM_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    index = VectorStoreIndex.from_documents(documents, show_progress=False)
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)
    return query_engine

if __name__ == "__main__":
    main() 
