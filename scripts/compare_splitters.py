#!/usr/bin/env python3
"""
Compare different text splitting methods to understand how they chunk documents.
This will show the differences between tree splitting and other approaches.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.indices.tree import TreeIndex
from llama_index.core.indices.vector_store import VectorStoreIndex

from scripts.load_document import load_document_llama_index
from config.settings import DOCUMENT_PATH, OUTPUT_DIR

def analyze_splitter(splitter_name: str, documents: List[Document], splitter_type: str = "sentence") -> Dict[str, Any]:
    """
    Analyze how a specific splitter chunks the documents.
    """
    print(f"\nAnalyzing {splitter_name}...")
    
    # Create appropriate splitter
    if splitter_type == "sentence":
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    elif splitter_type == "token":
        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    elif splitter_type == "tree":
        # For tree splitting, we'll analyze the tree index structure
        tree_index = TreeIndex.from_documents(documents)
        return analyze_tree_chunks(tree_index, splitter_name)
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    # Split documents
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Analyze chunks
    chunk_analysis = {
        "splitter_name": splitter_name,
        "splitter_type": splitter_type,
        "total_chunks": len(nodes),
        "chunks": []
    }
    
    total_chars = 0
    for i, node in enumerate(nodes):
        node_text = getattr(node, 'text', '')
        chunk_info = {
            "chunk_id": i,
            "text_length": len(node_text),
            "text_preview": node_text[:200] + "..." if len(node_text) > 200 else node_text,
            "word_count": len(node_text.split()),
            "sentence_count": len([s for s in node_text.split('.') if s.strip()])
        }
        chunk_analysis["chunks"].append(chunk_info)
        total_chars += len(node_text)
    
    chunk_analysis["total_characters"] = total_chars
    chunk_analysis["avg_chunk_size"] = total_chars / len(nodes) if nodes else 0
    
    return chunk_analysis

def analyze_tree_chunks(tree_index, splitter_name: str) -> Dict[str, Any]:
    """
    Analyze chunks from a tree index structure.
    """
    print(f"Analyzing tree index structure for {splitter_name}...")
    
    # For now, we'll do a basic analysis without deep tree traversal
    chunk_analysis = {
        "splitter_name": splitter_name,
        "splitter_type": "tree",
        "total_chunks": "Unknown (tree structure)",
        "chunks": [],
        "note": "Tree index structure analysis requires deeper inspection"
    }
    
    # Try to get basic information
    try:
        if hasattr(tree_index, 'docstore'):
            docstore = tree_index.docstore
            chunk_analysis["docstore_type"] = type(docstore).__name__
        
        if hasattr(tree_index, 'index_struct'):
            index_struct = tree_index.index_struct
            chunk_analysis["index_struct_type"] = type(index_struct).__name__
            
    except Exception as e:
        chunk_analysis["error"] = str(e)
    
    return chunk_analysis

def compare_splitters():
    """
    Compare different text splitting methods.
    """
    print("Loading document...")
    documents = load_document_llama_index(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s)")
    
    # Analyze different splitters
    splitters = [
        ("Sentence Splitter", "sentence"),
        ("Token Splitter", "token"),
        ("Tree Splitter", "tree")
    ]
    
    results = {}
    
    for splitter_name, splitter_type in splitters:
        try:
            analysis = analyze_splitter(splitter_name, documents, splitter_type)
            results[splitter_name] = analysis
        except Exception as e:
            print(f"Error analyzing {splitter_name}: {e}")
            results[splitter_name] = {"error": str(e)}
    
    # Save results
    comparison_file = os.path.join(OUTPUT_DIR, "splitter_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSplitter comparison saved to {comparison_file}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("SPLITTER COMPARISON SUMMARY")
    print("="*80)
    
    for splitter_name, analysis in results.items():
        if "error" in analysis:
            print(f"\n{splitter_name}: ERROR - {analysis['error']}")
            continue
            
        print(f"\n{splitter_name}:")
        print(f"  Total chunks: {analysis['total_chunks']}")
        
        if 'total_characters' in analysis:
            print(f"  Total characters: {analysis['total_characters']:,}")
            print(f"  Average chunk size: {analysis['avg_chunk_size']:.0f} characters")
        
        if analysis['splitter_type'] == 'tree':
            print(f"  Note: {analysis.get('note', 'Tree structure analysis')}")
    
    # Show chunk size distribution for non-tree splitters
    print("\n" + "="*80)
    print("CHUNK SIZE DISTRIBUTION")
    print("="*80)
    
    for splitter_name, analysis in results.items():
        if "error" in analysis or analysis['splitter_type'] == 'tree':
            continue
            
        print(f"\n{splitter_name}:")
        chunk_sizes = [c['text_length'] for c in analysis['chunks']]
        
        size_ranges = [
            ("Small (< 500 chars)", lambda x: x < 500),
            ("Medium (500-2000 chars)", lambda x: 500 <= x < 2000),
            ("Large (>= 2000 chars)", lambda x: x >= 2000)
        ]
        
        for range_name, condition in size_ranges:
            count = len([size for size in chunk_sizes if condition(size)])
            percentage = (count / len(chunk_sizes)) * 100 if chunk_sizes else 0
            print(f"  {range_name}: {count} chunks ({percentage:.1f}%)")
    
    # Show sample chunks from each splitter
    print("\n" + "="*80)
    print("SAMPLE CHUNKS FROM EACH SPLITTER")
    print("="*80)
    
    for splitter_name, analysis in results.items():
        if "error" in analysis or analysis['splitter_type'] == 'tree':
            continue
            
        print(f"\n{splitter_name} (showing first 3 chunks):")
        for i, chunk in enumerate(analysis['chunks'][:3]):
            print(f"  Chunk {i+1} ({chunk['text_length']} chars):")
            print(f"    {chunk['text_preview']}")
            print()

def main():
    """Main function to run the splitter comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting splitter comparison analysis...")
    print(f"Document: {DOCUMENT_PATH}")
    
    try:
        compare_splitters()
        print(f"\nComparison complete! Check {OUTPUT_DIR}/splitter_comparison.json for detailed data.")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 