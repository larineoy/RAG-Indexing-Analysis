"""
LangChain vectorstore example for RAG indexing analysis.
Uses Ollama LLM and HuggingFace embeddings.
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from scripts.load_document import load_document_langchain
from config.settings import DOCUMENT_PATH, LLM_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


def main():
    # 1. Load your data as a list of LangChain Document objects
    print(f"Loading document: {DOCUMENT_PATH}")
    documents = load_document_langchain(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")

    # 2. Set up LLM and embedding model
    llm = Ollama(model=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    print("Splitting documents into chunks...")
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")

    # 4. Build the Chroma vectorstore
    print("Building Chroma vectorstore (this may take a while for large documents)...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="outputs/langchain_vectorstore"
    )
    print("Chroma vectorstore built!")

    # 5. Query the vectorstore
    query = "What are the main specifications of the 5.56mm ammunition?"
    print(f"\nQuery: {query}")
    results = vectorstore.similarity_search(query, k=TOP_K)
    print("\nTop results:")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\n{doc.page_content[:500]}\n---")

    # 6. (Optional) Save the vectorstore for later use
    # vectorstore.persist()

def get_query_engine():
    documents = load_document_langchain(DOCUMENT_PATH)
    llm = Ollama(model=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="outputs/langchain_vectorstore"
    )
    def query_fn(query, k=TOP_K):
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    return query_fn

if __name__ == "__main__":
    main() 