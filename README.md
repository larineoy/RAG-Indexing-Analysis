# RAG Indexing Analysis

A comprehensive comparison framework for different RAG (Retrieval-Augmented Generation) indexing approaches. This project analyzes and compares the performance of LlamaIndex VectorStoreIndex, LlamaIndex TreeIndex, and LangChain Chroma implementations.

## 🎯 Project Overview

This project provides a systematic approach to evaluate different RAG indexing strategies using:
- **LLM**: Ollama (gemma3:4b)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Evaluation**: Timing, performance, and quality metrics

### Prerequisites

1. **Python 3.8+**
2. **Ollama** with gemma3:4b model

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG-Indexing-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull gemma3:4b
   ```

### Running the Analysis

#### Option 1: Full Comparison (Recommended)
```bash
python scripts/evaluate_and_time.py
```

#### Option 2: Individual Testing
```bash
# Test LlamaIndex VectorStoreIndex
python scripts/llama_vector_index.py

# Test LlamaIndex TreeIndex
python scripts/llama_tree_index.py

# Test LangChain
python scripts/langchain_index.py
```

#### Option 3: Interactive Notebook
```bash
jupyter notebook notebooks/comparison_demo.ipynb
```

## 📊 Configuration

Edit `config/settings.py` to customize:

```python
# Model configurations
LLM_MODEL = "gemma3:4b"  # Ollama model
EMBED_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embeddings

# Retrieval settings
TOP_K = 3

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### Performance Metrics
- **Indexing Time**: Time to create and build indexes
- **Query Response Time**: Average time per query
- **Memory Usage**: Index size and runtime memory
- **Response Quality**: Source retrieval and relevance

### Comparison Approaches

#### 1. LlamaIndex VectorStoreIndex
- **Use Case**: Large document collections, similarity search
- **Strengths**: Fast retrieval, semantic search
- **Best For**: General document Q&A

#### 2. LlamaIndex TreeIndex
- **Use Case**: Structured documents, hierarchical relationships
- **Strengths**: Document structure understanding
- **Best For**: Technical documents, manuals

#### 3. LangChain Chroma
- **Use Case**: General-purpose vector storage
- **Strengths**: Easy integration, flexible
- **Best For**: Prototyping, custom workflows

## 📚 API Reference

### Core Classes

#### LlamaVectorIndexer
```python
indexer = LlamaVectorIndexer(document_path="path/to/doc.docx")
stats = indexer.load_and_index_documents()
result = indexer.query("Your question?")
```

#### LlamaTreeIndexer
```python
indexer = LlamaTreeIndexer(document_path="path/to/doc.docx")
stats = indexer.load_and_index_documents()
result = indexer.query("Your question?")
```

#### LangChainIndexer
```python
indexer = LangChainIndexer(document_path="path/to/doc.docx")
stats = indexer.load_and_index_documents()
result = indexer.query("Your question?")
```

#### RAGEvaluator
```python
evaluator = RAGEvaluator()
comparison = evaluator.run_comparison(test_queries)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/jerryjliu/llama_index) for the indexing framework
- [LangChain](https://github.com/langchain-ai/langchain) for the RAG toolkit
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models

---

**Note**: This project is designed for research and comparison purposes. Results may vary based on hardware, document characteristics, and specific use cases.