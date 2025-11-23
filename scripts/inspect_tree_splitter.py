#!/usr/bin/env python3
"""
Inspect the tree splitter configuration in TreeIndex to understand what's actually happening.
"""

from llama_index.core import TreeIndex, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.indices.tree import TreeIndex
from llama_index.llms.ollama import Ollama
from scripts.load_document import load_document_llama_index
from config.settings import DOCUMENT_PATH, LLM_MODEL, EMBED_MODEL
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def inspect_default_tree_splitter():
    """Inspect what tree splitter is used by default."""
    print("="*80)
    print("INSPECTING TREE SPLITTER CONFIGURATION")
    print("="*80)
    
    # Load document
    documents = load_document_llama_index(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")
    
    # Set up LLM and embedding
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    
    print("\n1. Creating TreeIndex with default settings...")
    print("-" * 50)
    
    # Create TreeIndex with default settings
    index = TreeIndex.from_documents(documents, show_progress=True)
    
    print("\n2. Inspecting TreeIndex configuration...")
    print("-" * 50)
    
    # Inspect the index structure
    print(f"Index type: {type(index).__name__}")
    print(f"Index attributes: {[attr for attr in dir(index) if not attr.startswith('_')]}")
    
    # Check if there's a node parser
    if hasattr(index, 'node_parser'):
        print(f"\nNode parser: {type(index.node_parser).__name__}")
        print(f"Node parser attributes: {[attr for attr in dir(index.node_parser) if not attr.startswith('_')]}")
    else:
        print("\nNo explicit node parser found")
    
    # Check the index structure
    if hasattr(index, 'index_struct'):
        print(f"\nIndex structure: {type(index.index_struct).__name__}")
        print(f"Index structure attributes: {[attr for attr in dir(index.index_struct) if not attr.startswith('_')]}")
    
    # Check the docstore
    if hasattr(index, 'docstore'):
        print(f"\nDocstore: {type(index.docstore).__name__}")
        if hasattr(index.docstore, 'docs'):
            print(f"Number of documents in store: {len(index.docstore.docs)}")
            
            # Show first few documents
            print("\nFirst 3 documents in store:")
            for i, (doc_id, doc) in enumerate(list(index.docstore.docs.items())[:3]):
                doc_text = getattr(doc, 'text', 'No text')
                print(f"  Doc {i+1} (ID: {doc_id}): {len(doc_text)} chars")
                print(f"    Preview: {doc_text[:100]}...")
    
    return index

def test_custom_tree_splitter():
    """Test TreeIndex with explicit tree splitter configuration."""
    print("\n" + "="*80)
    print("TESTING CUSTOM TREE SPLITTER CONFIGURATION")
    print("="*80)
    
    # Load document
    documents = load_document_llama_index(DOCUMENT_PATH)
    
    # Set up LLM and embedding
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    
    print("\n1. Creating TreeIndex with explicit tree splitter...")
    print("-" * 50)
    
    # Try different splitter configurations
    splitters_to_test = [
        ("Sentence Splitter", SentenceSplitter(chunk_size=1024, chunk_overlap=20)),
        ("Token Splitter", TokenTextSplitter(chunk_size=512, chunk_overlap=50)),
    ]
    
    for splitter_name, splitter in splitters_to_test:
        print(f"\nTesting {splitter_name}...")
        try:
            # Create TreeIndex with explicit splitter
            index = TreeIndex.from_documents(
                documents, 
                node_parser=splitter,
                show_progress=True
            )
            
            print(f"  Successfully created TreeIndex with {splitter_name}")
            
            # Check the node parser
            if hasattr(index, 'node_parser'):
                print(f"  Node parser: {type(index.node_parser).__name__}")
            
            # Check docstore
            if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs'):
                print(f"  Documents in store: {len(index.docstore.docs)}")
            
        except Exception as e:
            print(f"  Error with {splitter_name}: {e}")

def explain_tree_splitter():
    """Explain what the tree splitter is and how it works."""
    print("\n" + "="*80)
    print("WHAT IS THE TREE SPLITTER?")
    print("="*80)
    
    explanation = {
        "what_is_tree_splitter": [
            "The 'tree splitter' in TreeIndex is NOT a traditional text splitter",
            "TreeIndex uses a hierarchical approach to organize document content",
            "It creates a tree structure where:",
            "  - Internal nodes contain summaries/overviews",
            "  - Leaf nodes contain the actual document chunks",
            "  - Parent nodes guide search to relevant children"
        ],
        "how_it_works": [
            "1. Document is initially split using a base splitter (sentence/token)",
            "2. LLM creates summaries of groups of chunks",
            "3. These summaries become internal nodes in the tree",
            "4. Original chunks become leaf nodes",
            "5. Tree structure is built recursively"
        ],
        "default_behavior": [
            "TreeIndex.from_documents() uses default settings:",
            "  - Base splitter: Usually sentence-based",
            "  - Chunk size: Default LlamaIndex settings",
            "  - Tree building: Automatic hierarchy creation",
            "  - Summarization: LLM creates node summaries"
        ],
        "vs_traditional_splitting": [
            "Traditional splitters: Create flat list of chunks",
            "Tree splitter: Creates hierarchical tree structure",
            "Traditional: Simple chunking by size/sentences",
            "Tree: Intelligent grouping and summarization"
        ]
    }
    
    print("What is the Tree Splitter?")
    print("-" * 30)
    for point in explanation["what_is_tree_splitter"]:
        print(point)
    
    print("\nHow Does It Work?")
    print("-" * 20)
    for point in explanation["how_it_works"]:
        print(point)
    
    print("\nDefault Behavior:")
    print("-" * 20)
    for point in explanation["default_behavior"]:
        print(point)
    
    print("\nTree vs Traditional Splitting:")
    print("-" * 30)
    for point in explanation["vs_traditional_splitting"]:
        print(point)

def main():
    """Main function to inspect tree splitter configuration."""
    print("TREE SPLITTER INSPECTION")
    print("Understanding how TreeIndex actually splits and organizes your document")
    
    try:
        # Inspect default configuration
        index = inspect_default_tree_splitter()
        
        # Test custom configurations
        test_custom_tree_splitter()
        
        # Explain what tree splitter is
        explain_tree_splitter()
        
        print(f"\n" + "="*80)
        print("INSPECTION COMPLETE!")
        print("="*80)
        print("This shows you:")
        print("✅ What splitter TreeIndex uses by default")
        print("✅ How the tree structure is created")
        print("✅ What the actual chunks look like")
        print("✅ How tree splitting differs from traditional splitting")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 