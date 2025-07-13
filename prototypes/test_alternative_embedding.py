#!/usr/bin/env python3
"""
Quick test of alternative embedding models to compare with Qwen3-Embedding
"""

import json
import time
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

def test_model(model_name: str, chunks_file: str):
    print(f"\n🔄 Testing {model_name}")
    print("-" * 50)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using MPS")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Load model
    start = time.time()
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"✅ Loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Load a few chunks for quick test
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)[:100]  # Just first 100 for speed
    
    chunk_texts = [chunk["document"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    # Compute embeddings
    start = time.time()
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    embed_time = time.time() - start
    print(f"📊 Encoded {len(chunks)} chunks in {embed_time:.2f}s")
    
    # Test query
    query = "How do I create a campaign?"
    start = time.time()
    query_embedding = model.encode([query])
    query_time = time.time() - start
    
    # Find similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]
    
    print(f"🔍 Query: '{query}' ({query_time*1000:.1f}ms)")
    print("Top matches:")
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        chunk_id = chunk_ids[idx]
        print(f"   {i+1}. {score:.3f} - {chunk_id}")

def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    # Test multiple models
    models = [
        "all-MiniLM-L6-v2",           # Popular baseline
        "all-mpnet-base-v2",          # Higher quality
        "sentence-transformers/all-MiniLM-L12-v2",  # Larger version
    ]
    
    print("🧪 Testing Alternative Embedding Models")
    print("=" * 60)
    
    for model in models:
        test_model(model, chunks_file)

if __name__ == "__main__":
    main()