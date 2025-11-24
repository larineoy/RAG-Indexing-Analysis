import os
import time
import json
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from scripts.llama_vector_index import get_query_engine as get_llama_vector_engine
from scripts.llama_tree_index import get_query_engine as get_llama_tree_engine
from scripts.langchain_index import get_query_engine as get_langchain_engine
from config.settings import DOCUMENT_PATH, OUTPUT_DIR, VISUALS_DIR, TOP_K

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, VISUALS_DIR), exist_ok=True)

# Example queries and reference answers for scoring
QUERIES = [
    "Does the 5.56mm FoA need new training?",
]
REFERENCE_ANSWERS = [
    "No, the 5.56mm FoA does not need new training.",
]

# Simple string similarity for relevance (can be replaced with better metric)
def simple_relevance_score(answer: str, reference: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, answer.lower(), reference.lower()).ratio()

# Helper to get file size in MB
def get_file_size_mb(path: str) -> float:
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

# Helper to get current memory usage in MB
def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Helper to count tokens (approximate by whitespace split)
def count_tokens(text: str) -> int:
    return len(text.split())

# Run and evaluate a single approach
def evaluate_approach(name: str, get_engine_func, output_json: str) -> Dict[str, Any]:
    print(f"\nEvaluating {name}...")
    start_mem = get_memory_usage_mb()
    start_time = time.time()
    query_engine = get_engine_func()
    build_time = time.time() - start_time
    end_mem = get_memory_usage_mb()
    mem_used = end_mem - start_mem

    results = []
    for q, ref in zip(QUERIES, REFERENCE_ANSWERS):
        # For LlamaIndex, query_engine.query(q) returns a response object
        # For LangChain, query_engine(q) returns a list of strings
        if name.startswith("LangChain"):
            answers = query_engine(q, k=TOP_K)
            answer = answers[0] if answers else ""
        else:
            response = query_engine.query(q)
            answer = str(response)
        relevance = simple_relevance_score(answer, ref)
        token_count = count_tokens(answer)
        results.append({
            "query": q,
            "answer": answer,
            "reference": ref,
            "relevance": relevance,
            "token_count": token_count
        })
    output_path = os.path.join(OUTPUT_DIR, output_json)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Index size: check the size of the outputs directory (simulate index file)
    index_size_mb = get_file_size_mb(output_path)

    # Aggregate metrics
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    total_tokens = sum(r["token_count"] for r in results)
    metrics = {
        "Index Name": name,
        "Index Build Time (s)": build_time,
        "Index Size (MB)": index_size_mb,
        "Memory Usage (MB)": mem_used,
        "Avg Relevance Score": avg_relevance,
        "Total Token Count": total_tokens
    }
    return metrics

# Evaluate all approaches
metrics_list = []
metrics_list.append(evaluate_approach("LlamaIndex Vector", get_llama_vector_engine, "llamaindex_flat.json"))
metrics_list.append(evaluate_approach("LlamaIndex Tree", get_llama_tree_engine, "llamaindex_tree.json"))
metrics_list.append(evaluate_approach("LangChain", get_langchain_engine, "langchain_flat.json"))

# Save metrics table
metrics_df = pd.DataFrame(metrics_list)
metrics_csv = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
metrics_json = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
metrics_df.to_json(metrics_json, orient="records", indent=2)
print(f"Saved evaluation metrics to {metrics_csv} and {metrics_json}")

# Generate and save visuals
sns.set(style="whitegrid")

# Barplot for build time
plt.figure(figsize=(8, 5))
sns.barplot(x="Index Name", y="Index Build Time (s)", data=metrics_df)
plt.title("Index Build Time")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, VISUALS_DIR, "build_time.png"))
plt.close()

# Barplot for index size
plt.figure(figsize=(8, 5))
sns.barplot(x="Index Name", y="Index Size (MB)", data=metrics_df)
plt.title("Index Size (MB)")
plt.ylabel("MB")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, VISUALS_DIR, "index_size.png"))
plt.close()

# Barplot for memory usage
plt.figure(figsize=(8, 5))
sns.barplot(x="Index Name", y="Memory Usage (MB)", data=metrics_df)
plt.title("Memory Usage (MB)")
plt.ylabel("MB")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, VISUALS_DIR, "memory_usage.png"))
plt.close()

# Barplot for relevance score
plt.figure(figsize=(8, 5))
sns.barplot(x="Index Name", y="Avg Relevance Score", data=metrics_df)
plt.title("Average Relevance Score")
plt.ylabel("Score (0-1)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, VISUALS_DIR, "relevance_score.png"))
plt.close()

# Barplot for token count
plt.figure(figsize=(8, 5))
sns.barplot(x="Index Name", y="Total Token Count", data=metrics_df)
plt.title("Total Token Count")
plt.ylabel("Tokens")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, VISUALS_DIR, "token_count.png"))
plt.close()

print(f"Saved all visuals to {os.path.join(OUTPUT_DIR, VISUALS_DIR)}")
 