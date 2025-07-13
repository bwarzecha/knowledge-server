#!/usr/bin/env python3
"""
Test Qwen3-Embedding with proper usage (query prompts)
"""

import json
import time
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

def test_qwen3_fixed():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔄 Loading Qwen3-Embedding on {device}")
    
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
    
    # Load first 50 chunks for quick test
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)[:50]
    
    chunk_texts = [chunk["document"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    print(f"📊 Computing embeddings for {len(chunks)} chunks...")
    start = time.time()
    # Documents don't need prompts
    document_embeddings = model.encode(chunk_texts, show_progress_bar=False)
    print(f"✅ Done in {time.time() - start:.2f}s")
    
    # Test queries with proper prompts
    test_queries = [
        "How do I create a campaign?",
        "What fields are in Campaign schema?",
        "How to update a campaign?",
    ]
    
    print("\n🔍 Testing queries with proper prompts:")
    print("-" * 50)
    
    for query in test_queries:
        start = time.time()
        # Use query prompt for Qwen3
        query_embedding = model.encode([query], prompt_name="query")
        
        # Compute similarities using model's similarity method
        similarities = model.similarity(query_embedding, document_embeddings)
        similarities = similarities.numpy()[0]  # Convert to numpy and get first row
        
        query_time = time.time() - start
        
        # Get top 3
        top_indices = similarities.argsort()[::-1][:3]
        
        print(f"\nQuery: '{query}' ({query_time*1000:.1f}ms)")
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            chunk_id = chunk_ids[idx]
            print(f"   {i+1}. {score:.3f} - {chunk_id}")

if __name__ == "__main__":
    test_qwen3_fixed()