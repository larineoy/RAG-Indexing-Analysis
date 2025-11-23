"""
Evaluation and timing script for comparing different RAG indexing approaches.
Compares LlamaIndex VectorStoreIndex, TreeIndex, and LangChain implementations.
"""

import time
import json
import os
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from scripts.llama_vector_index import main as llama_vector_main
from scripts.llama_tree_index import main as llama_tree_main
from scripts.langchain_index import main as langchain_main
from config.settings import DOCUMENT_PATH, OUTPUT_DIR


def evaluate_script(script_main_func, approach_name: str) -> Dict[str, Any]:
    print(f"Evaluating {approach_name}...")
    start = time.time()
    try:
        script_main_func()
        elapsed = time.time() - start
        print(f"{approach_name} completed in {elapsed:.2f} seconds.")
        return {"success": True, "elapsed": elapsed}
    except Exception as e:
        print(f"Error evaluating {approach_name}: {e}")
        return {"success": False, "error": str(e)}


def run_comparison():
    print("Starting RAG Indexing Comparison...")
    print(f"Document: {DOCUMENT_PATH}")
    print("-" * 50)

    results = {}
    results["llama_vector"] = evaluate_script(llama_vector_main, "LlamaIndex VectorStoreIndex")
    results["llama_tree"] = evaluate_script(llama_tree_main, "LlamaIndex TreeIndex")
    results["langchain"] = evaluate_script(langchain_main, "LangChain Chroma")

    # Save results
    timings_data = {
        "timestamp": datetime.now().isoformat(),
        "document": DOCUMENT_PATH,
        "results": results
    }
    with open("outputs/timings.json", "w") as f:
        json.dump(timings_data, f, indent=2)
    print(f"Results saved to outputs/timings.json")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for approach, data in results.items():
        print(f"\n{approach}:")
        if data.get("success"):
            print(f"  Elapsed time: {data['elapsed']:.2f}s")
        else:
            print(f"  Error: {data['error']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_comparison()
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main() 