#!/usr/bin/env python3
"""
Analyze and visualize tree indexing structure to understand how the tree splitter works.
This script will show the actual node structure, chunking behavior, and tree hierarchy.
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path

from scripts.load_document import load_document_llama_index
from llama_index.core import TreeIndex, Settings
from llama_index.llms.ollama import Ollama
from config.settings import DOCUMENT_PATH, OUTPUT_DIR, LLM_MODEL, EMBED_MODEL

def create_tree_index(documents):
    """Create a tree index from documents."""
    # Set up LLM and embedding model
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    # Note: We'll skip embedding model setup for now to avoid import issues
    
    # Build the TreeIndex
    print("Building TreeIndex...")
    index = TreeIndex.from_documents(documents, show_progress=True)
    print("TreeIndex built!")
    
    return index

def analyze_tree_structure():
    """
    Analyze the tree index structure and save detailed information about nodes and chunks.
    """
    print("Loading document and creating tree index...")
    
    # Load document
    documents = load_document_llama_index(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")
    
    # Create tree index
    tree_index = create_tree_index(documents)
    print("Tree index created successfully")
    
    # Analyze what we can access from the tree index
    tree_analysis = {
        "index_type": "TreeIndex",
        "document_count": len(documents),
        "index_attributes": {},
        "query_test": {}
    }
    
    # Try to access various attributes of the tree index
    try:
        # Get basic index information
        tree_analysis["index_attributes"]["has_docstore"] = hasattr(tree_index, 'docstore')
        tree_analysis["index_attributes"]["has_index_struct"] = hasattr(tree_index, 'index_struct')
        tree_analysis["index_attributes"]["has_storage_context"] = hasattr(tree_index, 'storage_context')
        
        # Try to get docstore information
        if hasattr(tree_index, 'docstore'):
            docstore = tree_index.docstore
            tree_analysis["index_attributes"]["docstore_type"] = type(docstore).__name__
            tree_analysis["index_attributes"]["docstore_attributes"] = [attr for attr in dir(docstore) if not attr.startswith('_')]
        
        # Try to get index structure information
        if hasattr(tree_index, 'index_struct'):
            index_struct = tree_index.index_struct
            tree_analysis["index_attributes"]["index_struct_type"] = type(index_struct).__name__
            tree_analysis["index_attributes"]["index_struct_attributes"] = [attr for attr in dir(index_struct) if not attr.startswith('_')]
        
    except Exception as e:
        tree_analysis["index_attributes"]["error"] = str(e)
    
    # Test query functionality
    try:
        query_engine = tree_index.as_query_engine()
        test_query = "What is this document about?"
        response = query_engine.query(test_query)
        
        tree_analysis["query_test"]["success"] = True
        tree_analysis["query_test"]["response"] = str(response)
        tree_analysis["query_test"]["response_length"] = len(str(response))
        
    except Exception as e:
        tree_analysis["query_test"]["success"] = False
        tree_analysis["query_test"]["error"] = str(e)
    
    # Save analysis
    analysis_file = os.path.join(OUTPUT_DIR, "tree_structure_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(tree_analysis, f, indent=2, default=str)
    
    print(f"Tree structure analysis saved to {analysis_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TREE STRUCTURE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Index type: {tree_analysis['index_type']}")
    print(f"Document count: {tree_analysis['document_count']}")
    print(f"Has docstore: {tree_analysis['index_attributes'].get('has_docstore', False)}")
    print(f"Has index_struct: {tree_analysis['index_attributes'].get('has_index_struct', False)}")
    print(f"Has storage_context: {tree_analysis['index_attributes'].get('has_storage_context', False)}")
    
    if 'index_struct_type' in tree_analysis['index_attributes']:
        print(f"Index struct type: {tree_analysis['index_attributes']['index_struct_type']}")
    
    if 'query_test' in tree_analysis and tree_analysis['query_test'].get('success', False):
        print(f"Query test successful: {tree_analysis['query_test']['response_length']} characters")
        print(f"Sample response: {tree_analysis['query_test']['response'][:200]}...")
    else:
        print("Query test failed")
    
    return tree_analysis

def create_tree_visualization(tree_analysis):
    """
    Create a text-based tree visualization.
    """
    print("\n" + "="*60)
    print("TREE STRUCTURE VISUALIZATION")
    print("="*60)
    
    def print_tree(node_data, prefix="", is_last=True):
        """Print tree structure in a hierarchical format."""
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}Node {node_data['id']} (Depth {node_data['depth']})")
        
        if node_data['is_leaf']:
            print(f"{prefix}{'    ' if is_last else '│   '}  📄 {node_data['text_preview']}")
        else:
            print(f"{prefix}{'    ' if is_last else '│   '}  📁 Internal Node")
        
        children = node_data['children']
        for i, child in enumerate(children):
            print_tree(child, prefix + ('    ' if is_last else '│   '), i == len(children) - 1)
    
    print_tree(tree_analysis['tree_structure'])

def analyze_chunking_behavior(tree_analysis):
    """
    Analyze how the tree splitter is chunking the content.
    """
    print("\n" + "="*60)
    print("CHUNKING BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Analyze text length distribution
    text_lengths = [node['text_length'] for node in tree_analysis['node_details']]
    
    print(f"Text length statistics:")
    print(f"  Min: {min(text_lengths)} characters")
    print(f"  Max: {max(text_lengths)} characters")
    print(f"  Average: {sum(text_lengths) / len(text_lengths):.0f} characters")
    print(f"  Median: {sorted(text_lengths)[len(text_lengths)//2]} characters")
    
    # Analyze chunking patterns
    print(f"\nChunking patterns:")
    print(f"  Small chunks (< 500 chars): {len([l for l in text_lengths if l < 500])}")
    print(f"  Medium chunks (500-2000 chars): {len([l for l in text_lengths if 500 <= l < 2000])}")
    print(f"  Large chunks (>= 2000 chars): {len([l for l in text_lengths if l >= 2000])}")
    
    # Show examples of different chunk sizes
    print(f"\nExample chunks by size:")
    
    small_chunks = [n for n in tree_analysis['node_details'] if n['text_length'] < 500]
    if small_chunks:
        print(f"\nSmall chunk example:")
        print(f"  {small_chunks[0]['text_preview']}")
    
    large_chunks = [n for n in tree_analysis['node_details'] if n['text_length'] >= 2000]
    if large_chunks:
        print(f"\nLarge chunk example:")
        print(f"  {large_chunks[0]['text_preview']}")

def main():
    """Main function to run the tree structure analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting tree structure analysis...")
    print(f"Document: {DOCUMENT_PATH}")
    
    try:
        # Analyze tree structure
        tree_analysis = analyze_tree_structure()
        
        # Create visualization
        create_tree_visualization(tree_analysis)
        
        # Analyze chunking behavior
        analyze_chunking_behavior(tree_analysis)
        
        print(f"\nAnalysis complete! Check {OUTPUT_DIR}/tree_structure_analysis.json for detailed data.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
