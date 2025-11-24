#!/usr/bin/env python3
"""
Show the actual chunks/nodes created by the tree splitter during indexing.
This will reveal what the tree splitter is actually splitting on and how it organizes the content.
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path

from scripts.load_document import load_document_llama_index, get_document_text
from llama_index.core import TreeIndex, Settings
from llama_index.llms.ollama import Ollama
from config.settings import DOCUMENT_PATH, OUTPUT_DIR, LLM_MODEL

def show_document_structure():
    """First, show the original document structure to understand what we're working with."""
    print("="*80)
    print("STEP 1: ORIGINAL DOCUMENT STRUCTURE")
    print("="*80)
    
    documents = load_document_llama_index(DOCUMENT_PATH)
    content = get_document_text(documents)
    
    print(f"Document: {DOCUMENT_PATH}")
    print(f"Total content: {len(content):,} characters")
    print(f"Total words: {len(content.split()):,}")
    
    # Show document structure
    lines = content.split('\n')
    print(f"\nDocument has {len(lines)} lines")
    
    # Look for section markers
    print("\nLooking for section markers in the document:")
    print("-" * 50)
    
    section_markers = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            # Look for potential section headers
            if (line.isupper() or 
                line.endswith(':') or 
                line.startswith('Section') or
                line.startswith('CHAPTER') or
                line.startswith('PART') or
                len(line) < 100 and line[0].isupper() and line.isupper()):
                section_markers.append((i, line))
    
    if section_markers:
        print(f"Found {len(section_markers)} potential section markers:")
        for line_num, marker in section_markers[:20]:  # Show first 20
            print(f"  Line {line_num}: {marker}")
        if len(section_markers) > 20:
            print(f"  ... and {len(section_markers) - 20} more")
    else:
        print("No obvious section markers found")
    
    # Show first 500 characters to understand content
    print(f"\nDocument preview (first 500 characters):")
    print("-" * 50)
    print(content[:500])
    print("-" * 50)
    
    return documents, content, section_markers

def create_tree_index_and_analyze(documents):
    """Create the tree index and analyze what chunks/nodes are created."""
    print("\n" + "="*80)
    print("STEP 2: CREATING TREE INDEX AND ANALYZING CHUNKS")
    print("="*80)
    
    # Set up LLM
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    
    print("Building TreeIndex...")
    print("This will show you exactly what chunks/nodes are created during indexing")
    print("-" * 50)
    
    # Create the tree index
    tree_index = TreeIndex.from_documents(documents, show_progress=True)
    print("TreeIndex built!")
    
    return tree_index

def analyze_tree_nodes(tree_index):
    """Analyze the actual nodes/chunks in the tree index."""
    print("\n" + "="*80)
    print("STEP 3: ANALYZING TREE NODES/CHUNKS")
    print("="*80)
    
    node_analysis = {
        "total_nodes": 0,
        "leaf_nodes": 0,
        "internal_nodes": 0,
        "node_details": [],
        "chunk_sizes": [],
        "content_samples": []
    }
    
    # Try to access the docstore to see all nodes
    if hasattr(tree_index, 'docstore') and hasattr(tree_index.docstore, 'docs'):
        docstore = tree_index.docstore
        all_docs = docstore.docs
        
        print(f"Found {len(all_docs)} documents/nodes in the tree index")
        print("-" * 50)
        
        node_analysis["total_nodes"] = len(all_docs)
        
        # Analyze each node
        for i, (doc_id, doc) in enumerate(all_docs.items()):
            doc_text = getattr(doc, 'text', 'No text')
            doc_type = getattr(doc, 'node_type', 'Unknown')
            
            node_info = {
                "node_id": doc_id,
                "node_type": doc_type,
                "text_length": len(doc_text),
                "word_count": len(doc_text.split()),
                "text_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                "full_text": doc_text
            }
            
            node_analysis["node_details"].append(node_info)
            node_analysis["chunk_sizes"].append(len(doc_text))
            
            # Categorize nodes
            if hasattr(doc, 'is_leaf'):
                if doc.is_leaf:
                    node_analysis["leaf_nodes"] += 1
                else:
                    node_analysis["internal_nodes"] += 1
            
            # Show first few nodes in detail
            if i < 5:
                print(f"\nNode {i+1} (ID: {doc_id}):")
                print(f"  Type: {doc_type}")
                print(f"  Length: {len(doc_text)} characters")
                print(f"  Words: {len(doc_text.split())}")
                print(f"  Content: {doc_text[:150]}...")
                print("-" * 30)
        
        # Show summary statistics
        print(f"\n" + "="*50)
        print("NODE ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total nodes: {node_analysis['total_nodes']}")
        print(f"Leaf nodes: {node_analysis['leaf_nodes']}")
        print(f"Internal nodes: {node_analysis['internal_nodes']}")
        
        if node_analysis["chunk_sizes"]:
            print(f"Average chunk size: {sum(node_analysis['chunk_sizes']) / len(node_analysis['chunk_sizes']):.0f} characters")
            print(f"Min chunk size: {min(node_analysis['chunk_sizes'])} characters")
            print(f"Max chunk size: {max(node_analysis['chunk_sizes'])} characters")
        
        # Analyze chunk size distribution
        print(f"\nChunk size distribution:")
        small_chunks = [size for size in node_analysis["chunk_sizes"] if size < 500]
        medium_chunks = [size for size in node_analysis["chunk_sizes"] if 500 <= size < 2000]
        large_chunks = [size for size in node_analysis["chunk_sizes"] if size >= 2000]
        
        print(f"  Small chunks (< 500 chars): {len(small_chunks)}")
        print(f"  Medium chunks (500-2000 chars): {len(medium_chunks)}")
        print(f"  Large chunks (>= 2000 chars): {len(large_chunks)}")
    
    else:
        print("Could not access docstore to analyze nodes")
        node_analysis["error"] = "Could not access docstore"
    
    return node_analysis

def test_tree_query_behavior(tree_index):
    """Test how the tree responds to queries to understand the chunking."""
    print("\n" + "="*80)
    print("STEP 4: TESTING TREE QUERY BEHAVIOR")
    print("="*80)
    
    query_engine = tree_index.as_query_engine()
    
    # Test queries to see how the tree responds
    test_queries = [
        "What is this document about?",
        "What are the main sections or topics?",
        "What are the key specifications or details?",
        "What are the safety considerations?",
        "What are the performance characteristics?"
    ]
    
    query_results = {}
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        try:
            response = query_engine.query(query)
            response_text = str(response)
            
            print(f"Response length: {len(response_text)} characters")
            print(f"Response: {response_text[:300]}...")
            
            query_results[query] = {
                "response_length": len(response_text),
                "response": response_text,
                "success": True
            }
            
        except Exception as e:
            print(f"Error: {e}")
            query_results[query] = {
                "error": str(e),
                "success": False
            }
    
    return query_results

def evaluate_section_splitting(node_analysis, section_markers):
    """Evaluate whether the tree splitter is correctly splitting on sections."""
    print("\n" + "="*80)
    print("STEP 5: EVALUATING SECTION SPLITTING")
    print("="*80)
    
    evaluation = {
        "section_markers_found": len(section_markers),
        "nodes_created": node_analysis.get("total_nodes", 0),
        "splitting_quality": "Unknown"
    }
    
    print(f"Original document had {len(section_markers)} potential section markers")
    print(f"Tree splitter created {node_analysis.get('total_nodes', 0)} nodes")
    
    # Analyze if nodes correspond to sections
    if node_analysis.get("node_details"):
        print(f"\nAnalyzing if nodes correspond to document sections:")
        print("-" * 50)
        
        # Look for section markers in node content
        nodes_with_sections = 0
        for node in node_analysis["node_details"]:
            node_text = node["full_text"]
            
            # Check if this node contains any section markers
            contains_section = False
            for line_num, marker in section_markers:
                if marker.lower() in node_text.lower():
                    contains_section = True
                    break
            
            if contains_section:
                nodes_with_sections += 1
                print(f"  Node {node['node_id']} contains section markers")
        
        print(f"\nNodes containing section markers: {nodes_with_sections}")
        
        # Evaluate splitting quality
        if nodes_with_sections > 0:
            evaluation["splitting_quality"] = "Good - nodes contain section markers"
            print("✅ Tree splitter appears to be respecting document sections")
        else:
            evaluation["splitting_quality"] = "Poor - no section markers found in nodes"
            print("❌ Tree splitter may not be respecting document sections")
    
    return evaluation

def main():
    """Main function to show tree chunks and evaluate splitting."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("TREE SPLITTER CHUNK ANALYSIS")
    print("This will show you exactly what chunks/nodes the tree splitter creates")
    print("and whether it's correctly splitting on sections and subsections")
    print("="*80)
    
    try:
        # Step 1: Show document structure
        documents, content, section_markers = show_document_structure()
        
        # Step 2: Create tree index
        tree_index = create_tree_index_and_analyze(documents)
        
        # Step 3: Analyze tree nodes
        node_analysis = analyze_tree_nodes(tree_index)
        
        # Step 4: Test query behavior
        query_results = test_tree_query_behavior(tree_index)
        
        # Step 5: Evaluate section splitting
        splitting_evaluation = evaluate_section_splitting(node_analysis, section_markers)
        
        # Save comprehensive analysis
        full_analysis = {
            "document_info": {
                "path": DOCUMENT_PATH,
                "content_length": len(content),
                "section_markers": section_markers
            },
            "node_analysis": node_analysis,
            "query_results": query_results,
            "splitting_evaluation": splitting_evaluation
        }
        
        analysis_file = os.path.join(OUTPUT_DIR, "tree_chunk_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Detailed analysis saved to: {analysis_file}")
        print("\nThis analysis shows you:")
        print("✅ What chunks/nodes the tree splitter creates")
        print("✅ How it splits your document")
        print("✅ Whether it respects sections and subsections")
        print("✅ How the tree responds to queries")
        print("✅ The quality of the splitting")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 