# Configuration settings for RAG Indexing Analysis

# Document paths
DOCUMENT_PATH = "data/5 56mm FoA.docx"
CORPUS_DIR = "data/"

# Model configurations
LLM_MODEL = "gemma3:4b"  # Ollama model
EMBED_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embeddings

# Retrieval settings
TOP_K = 3

# Output paths
OUTPUT_DIR = "outputs/"
VISUALS_DIR = "visuals/"
REPORTS_DIR = "reports/"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50 