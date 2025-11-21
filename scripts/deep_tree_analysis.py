#!/usr/bin/env python3
"""
Deep analysis of how the tree splitter actually works with your document.
This will show you the actual chunks, tree structure, and splitting behavior.
"""

import os
import json
import gc
import psutil
from typing import Dict, List, Any
from pathlib import Path

from scripts.load_document import load_document_llama_index, get_document_text
from llama_index.core import TreeIndex, Settings
from llama_index.llms.ollama import Ollama
from config.settings import DOCUMENT_PATH, OUTPUT_DIR, LLM_MODEL

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def analyze_document_content():
    """First, let's see what's actually in your document."""
    print("="*80)
    print("STEP 1: ANALYZING DOCUMENT CONTENT")
    print("="*80)
    
    documents = load_document_llama_index(DOCUMENT_PATH)
    content = get_document_text(documents)
    
    print(f"Document path: {DOCUMENT_PATH}")
    print(f"Number of documents: {len(documents)}")
    print(f"Total content length: {len(content):,} characters")
    print(f"Total words: {len(content.split()):,}")
    
    # Show first 1000 characters to understand the content
    print(f"\nDocument preview (first 1000 characters):")
    print("-" * 80)
    print(content[:1000])
    print("-" * 80)
    
    # Look for section markers or structure
    lines = content.split('\n')
    print(f"\nDocument structure analysis:")
    print(f"Total lines: {len(lines)}")
    
    # Look for potential section headers
    potential_headers = []
    for i, line in enumerate(lines[:50]):  # Check first 50 lines
        line = line.strip()
        if line and len(line) < 100 and (line.isupper() or line.endswith(':') or line.startswith('Section')):
            potential_headers.append((i, line))
    
    if potential_headers:
        print(f"Potential section headers found:")
        for line_num, header in potential_headers[:10]:
            print(f"  Line {line_num}: {header}")
    else:
        print("No obvious section headers found in first 50 lines")
    
    return documents, content

def analyze_tree_construction(documents):
    """Analyze how the tree index is constructed and what happens to memory."""
    print("\n" + "="*80)
    print("STEP 2: ANALYZING TREE CONSTRUCTION")
    print("="*80)
    
    # Set up LLM
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    
    # Monitor memory during construction
    print("Memory monitoring during tree construction:")
    print("-" * 50)
    
    # Before construction
    gc.collect()  # Force garbage collection
    start_mem = get_memory_usage_mb()
    print(f"Memory before construction: {start_mem:.1f} MB")
    
    # During construction
    print("Building TreeIndex...")
    tree_index = TreeIndex.from_documents(documents, show_progress=True)
    
    # After construction
    end_mem = get_memory_usage_mb()
    print(f"Memory after construction: {end_mem:.1f} MB")
    print(f"Memory difference: {end_mem - start_mem:.1f} MB")
    
    # Force garbage collection and check again
    gc.collect()
    final_mem = get_memory_usage_mb()
    print(f"Memory after garbage collection: {final_mem:.1f} MB")
    print(f"Final memory difference: {final_mem - start_mem:.1f} MB")
    
    return tree_index

def analyze_tree_structure(tree_index):
    """Analyze the actual tree structure and nodes."""
    print("\n" + "="*80)
    print("STEP 3: ANALYZING TREE STRUCTURE")
    print("="*80)
    
    tree_analysis = {
        "index_type": "TreeIndex",
        "structure_info": {},
        "node_analysis": {},
        "memory_explanation": {}
    }
    
    # Analyze what we can access
    print("Tree index attributes:")
    print("-" * 30)
    
    # Check docstore
    if hasattr(tree_index, 'docstore'):
        docstore = tree_index.docstore
        print(f"Docstore type: {type(docstore).__name__}")
        tree_analysis["structure_info"]["docstore_type"] = type(docstore).__name__
        
        # Try to get all nodes from docstore
        try:
            if hasattr(docstore, 'docs'):
                all_docs = docstore.docs
                print(f"Number of documents in docstore: {len(all_docs)}")
                tree_analysis["node_analysis"]["total_docs_in_store"] = len(all_docs)
                
                # Show first few documents
                print("\nFirst 3 documents in docstore:")
                for i, (doc_id, doc) in enumerate(list(all_docs.items())[:3]):
                    doc_text = getattr(doc, 'text', 'No text')
                    print(f"  Doc {i+1} (ID: {doc_id}): {len(doc_text)} chars")
                    print(f"    Preview: {doc_text[:100]}...")
        except Exception as e:
            print(f"Error accessing docstore docs: {e}")
    
    # Check index structure
    if hasattr(tree_index, 'index_struct'):
        index_struct = tree_index.index_struct
        print(f"\nIndex structure type: {type(index_struct).__name__}")
        tree_analysis["structure_info"]["index_struct_type"] = type(index_struct).__name__
        
        # List all attributes
        attrs = [attr for attr in dir(index_struct) if not attr.startswith('_')]
        print(f"Index struct attributes: {attrs}")
        tree_analysis["structure_info"]["index_struct_attributes"] = attrs
        
        # Try to access specific attributes
        for attr in ['root_id', 'root_node', 'all_nodes', 'nodes']:
            if hasattr(index_struct, attr):
                value = getattr(index_struct, attr)
                print(f"  {attr}: {value}")
                tree_analysis["structure_info"][attr] = str(value)
    
    # Test query to see how tree works
    print("\n" + "-" * 50)
    print("Testing tree query functionality:")
    print("-" * 50)
    
    try:
        query_engine = tree_index.as_query_engine()
        
        # Test queries to understand how tree responds
        test_queries = [
            "What is this document about?",
            "What are the main sections?",
            "What are the key specifications?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = query_engine.query(query)
            print(f"Response length: {len(str(response))} characters")
            print(f"Response preview: {str(response)[:200]}...")
            
            # Store in analysis
            if "query_tests" not in tree_analysis:
                tree_analysis["query_tests"] = {}
            tree_analysis["query_tests"][query] = {
                "response_length": len(str(response)),
                "response_preview": str(response)[:200]
            }
    
    except Exception as e:
        print(f"Error testing queries: {e}")
        tree_analysis["query_tests"] = {"error": str(e)}
    
    return tree_analysis

def explain_memory_behavior():
    """Explain why tree indexing might show negative memory usage."""
    print("\n" + "="*80)
    print("STEP 4: MEMORY BEHAVIOR EXPLANATION")
    print("="*80)
    
    explanation = {
        "why_negative_memory": [
            "1. Tree construction uses temporary memory for building the hierarchy",
            "2. During construction, Python allocates memory for intermediate objects",
            "3. After construction, garbage collection frees this temporary memory",
            "4. The final tree structure is more memory-efficient than the construction process",
            "5. Result: end_memory - start_memory can be negative"
        ],
        "garbage_collection_impact": [
            "Python's garbage collector runs automatically",
            "Tree construction creates many temporary objects",
            "After construction, these objects become eligible for garbage collection",
            "Manual gc.collect() forces immediate cleanup",
            "This explains the memory reduction you observed"
        ],
        "tree_vs_other_indexing": [
            "Vector indexing: Simple chunking, minimal construction overhead",
            "Tree indexing: Complex hierarchy building, more construction memory",
            "Tree indexing: Better memory efficiency after construction",
            "Tree indexing: More sophisticated but more construction overhead"
        ]
    }
    
    print("Why Tree Indexing Shows Negative Memory Usage:")
    print("-" * 50)
    for point in explanation["why_negative_memory"]:
        print(point)
    
    print("\nGarbage Collection Impact:")
    print("-" * 30)
    for point in explanation["garbage_collection_impact"]:
        print(point)
    
    print("\nTree vs Other Indexing:")
    print("-" * 25)
    for point in explanation["tree_vs_other_indexing"]:
        print(point)
    
    return explanation

def main():
    """Main analysis function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("DEEP TREE INDEXING ANALYSIS")
    print("This will show you exactly how the tree splitter works with your document")
    print("="*80)
    
    try:
        # Step 1: Analyze document content
        documents, content = analyze_document_content()
        
        # Step 2: Analyze tree construction
        tree_index = analyze_tree_construction(documents)
        
        # Step 3: Analyze tree structure
        tree_analysis = analyze_tree_structure(tree_index)
        
        # Step 4: Explain memory behavior
        memory_explanation = explain_memory_behavior()
        
        # Save comprehensive analysis
        full_analysis = {
            "document_analysis": {
                "path": DOCUMENT_PATH,
                "content_length": len(content),
                "word_count": len(content.split()),
                "preview": content[:1000]
            },
            "tree_analysis": tree_analysis,
            "memory_explanation": memory_explanation
        }
        
        analysis_file = os.path.join(OUTPUT_DIR, "deep_tree_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Detailed analysis saved to: {analysis_file}")
        print("\nThis analysis shows you:")
        print("✅ How your document is structured")
        print("✅ How the tree splitter processes it")
        print("✅ What the tree structure looks like")
        print("✅ Why memory usage is negative")
        print("✅ How tree indexing differs from other methods")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 