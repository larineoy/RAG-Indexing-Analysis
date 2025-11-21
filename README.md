# RAG Indexing Analysis

A comprehensive comparison framework for different RAG (Retrieval-Augmented Generation) indexing approaches. This project analyzes and compares the performance of LlamaIndex VectorStoreIndex, LlamaIndex TreeIndex, and LangChain Chroma implementations.

## 🎯 Project Overview

This project provides a systematic approach to evaluate different RAG indexing strategies using:
- **LLM**: Ollama (gemma3:4b)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Document**: 5.56mm FoA.docx
- **Evaluation**: Timing, performance, and quality metrics

## 📁 Project Structure

```
RAG-Indexing-Analysis/
├── 📁 data/
│   └── 5 56mm FoA.docx                          # Input document for indexing
├── 📁 scripts/
│   ├── load_document.py                         # Shared loader functions
│   ├── llama_vector_index.py                    # LlamaIndex VectorStoreIndex
│   ├── llama_tree_index.py                      # LlamaIndex TreeIndex
│   ├── langchain_index.py                       # LangChain vectorstore
│   └── evaluate_and_time.py                     # Evaluation framework
├── 📁 outputs/
│   ├── llama_vector_response.txt                # Llama VectorStoreIndex output
│   ├── llama_tree_response.txt                  # Llama TreeIndex output
│   ├── langchain_response.txt                   # LangChain output
│   └── timings.json                             # Performance metrics
├── 📁 visuals/
│   └── comparison_results.png                   # Performance charts
├── 📁 notebooks/
│   └── comparison_demo.ipynb                    # Interactive analysis
├── 📁 config/
│   └── settings.py                              # Configuration settings
├── 📁 reports/
│   └── final_report.md                          # Analysis report
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** with gemma3:4b model
3. **Document**: Place `5 56mm FoA.docx` in the `data/` directory

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

4. **Add your document**:
   ```bash
   # Place your document in the data directory
   cp "path/to/your/5 56mm FoA.docx" data/
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
# Document paths
DOCUMENT_PATH = "data/5 56mm FoA.docx"
CORPUS_DIR = "data/"

# Model configurations
LLM_MODEL = "gemma3:4b"  # Ollama model
EMBED_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embeddings

# Retrieval settings
TOP_K = 3

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

## 🔍 Analysis Features

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

## 📈 Results Interpretation

### Performance Comparison
The framework generates:
- **Timing Data**: JSON format with detailed metrics
- **Response Files**: Text files with query responses
- **Visualizations**: Charts comparing performance
- **Reports**: Markdown reports with analysis

### Key Metrics to Consider
1. **Indexing Speed**: How quickly documents are processed
2. **Query Latency**: Response time for user queries
3. **Memory Efficiency**: Resource usage patterns
4. **Response Quality**: Relevance and accuracy of answers

## 🛠️ Customization

### Adding New Documents
1. Place documents in `data/` directory
2. Update `DOCUMENT_PATH` in `config/settings.py`
3. Run the evaluation scripts

### Modifying Test Queries
Edit `get_test_queries()` in `scripts/evaluate_and_time.py`:

```python
def get_test_queries() -> List[str]:
    return [
        "Your custom query 1?",
        "Your custom query 2?",
        # Add more queries...
    ]
```

### Adding New Indexing Approaches
1. Create new script in `scripts/` directory
2. Implement the same interface as existing indexers
3. Add evaluation method to `RAGEvaluator` class
4. Update the main comparison function

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

#### 2. Model Not Found
```bash
# Pull the required model
ollama pull gemma3:4b
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 4. Document Loading Issues
- Ensure document is in supported format (DOCX, PDF, TXT)
- Check file permissions
- Verify document path in settings

### Debug Mode
Enable verbose logging by modifying the scripts to include:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black scripts/ config/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/jerryjliu/llama_index) for the indexing framework
- [LangChain](https://github.com/langchain-ai/langchain) for the RAG toolkit
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This project is designed for research and comparison purposes. Results may vary based on hardware, document characteristics, and specific use cases.