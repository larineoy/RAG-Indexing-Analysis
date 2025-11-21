#!/usr/bin/env python3
"""
Temporary script to extract content from the document
to evaluate if the evaluation questions are applicable.
"""

from scripts.load_document import load_document_llama_index, get_document_text
from config.settings import DOCUMENT_PATH

def main():
    print("Extracting content from document...")
    print(f"Document path: {DOCUMENT_PATH}")
    print("-" * 80)
    
    try:
        # Load the document
        documents = load_document_llama_index(DOCUMENT_PATH)
        
        # Extract text content
        content = get_document_text(documents)
        
        # Display first 2000 characters to see what the document contains
        print("Document content (first 2000 characters):")
        print("=" * 80)
        print(content[:2000])
        print("=" * 80)
        
        # Display document statistics
        print(f"\nDocument statistics:")
        print(f"Number of documents: {len(documents)}")
        print(f"Total content length: {len(content)} characters")
        print(f"Total words: {len(content.split())}")
        
        # Check for key terms that would indicate if the questions are relevant
        key_terms = [
            "specification", "specifications", "caliber", "weight", "velocity",
            "range", "effective", "safety", "handling", "storage",
            "performance", "characteristics", "temperature", "pressure"
        ]
        
        print(f"\nKey terms found in document:")
        content_lower = content.lower()
        for term in key_terms:
            count = content_lower.count(term)
            if count > 0:
                print(f"  '{term}': {count} occurrences")
        
    except Exception as e:
        print(f"Error extracting content: {e}")

if __name__ == "__main__":
    main() 
