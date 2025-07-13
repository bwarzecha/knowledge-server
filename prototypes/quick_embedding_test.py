#!/usr/bin/env python3
"""
Quick validation with all-MiniLM-L6-v2 to see if embeddings can work
"""

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

def quick_test():
    # Load chunks
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Setup model with MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔄 Loading all-MiniLM-L6-v2 on {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    chunk_texts = [chunk["document"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    print(f"📊 Computing embeddings for {len(chunks)} chunks...")
    start = time.time()
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    print(f"✅ Done in {time.time() - start:.2f}s")
    
    # Test a few queries
    test_queries = [
        "How do I create a campaign?",
        "What fields are in Campaign schema?",
        "How to update a campaign?",
        "What does error 400 mean?",
        "List campaigns endpoint"
    ]
    
    print("\n🔍 Testing queries:")
    print("-" * 50)
    
    for query in test_queries:
        start = time.time()
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[::-1][:3]
        query_time = time.time() - start
        
        print(f"\nQuery: '{query}' ({query_time*1000:.1f}ms)")
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            chunk_id = chunk_ids[idx]
            print(f"   {i+1}. {score:.3f} - {chunk_id}")

if __name__ == "__main__":
    quick_test()