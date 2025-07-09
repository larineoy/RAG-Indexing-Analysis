#!/usr/bin/env python3
"""
Run tree structure analysis and splitter comparison to understand how tree indexing works.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*80)
    print("TREE INDEXING ANALYSIS")
    print("="*80)
    
    print("\n1. Running tree structure analysis...")
    from scripts.analyze_tree_structure import main as analyze_main
    analyze_main()
    
    print("\n" + "="*80)
    print("\n2. Running splitter comparison...")
    from scripts.compare_splitters import main as compare_main
    compare_main()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nCheck the outputs/ directory for detailed analysis files:")
    print("- tree_structure_analysis.json: Detailed tree structure")
    print("- splitter_comparison.json: Comparison of different splitters")
    print("\nThis will help you understand:")
    print("- How the tree splitter chunks your document")
    print("- What the tree structure looks like")
    print("- How it compares to other splitting methods")
    print("- Why memory usage might be negative (garbage collection during construction)")

if __name__ == "__main__":
    main() 