#!/usr/bin/env python3
"""
Test Qwen3-Embedding-8B-GGUF with 4-bit quantization for better performance
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import hf_hub_download
import requests


@dataclass
class QueryTest:
    query: str
    expected_chunks: List[str]
    expected_types: List[str]


def download_gguf_model():
    """Download the GGUF model file."""
    print("🔄 Downloading Qwen3-Embedding-8B GGUF model...")
    
    try:
        # Download the Q4_K_M quantized version (good balance of quality/size)
        # Let's first check what files are available
        from huggingface_hub import list_repo_files
        
        files = list_repo_files("Qwen/Qwen3-Embedding-8B-GGUF")
        gguf_files = [f for f in files if f.endswith('.gguf')]
        print(f"Available GGUF files: {gguf_files}")
        
        # Use the first available GGUF file (likely q4_k_m variant)
        if gguf_files:
            filename = gguf_files[0]
            print(f"Using file: {filename}")
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen3-Embedding-8B-GGUF",
                filename=filename,
                cache_dir="./model_cache"
            )
        else:
            raise ValueError("No GGUF files found in repository")
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def test_gguf_embedding():
    """Test the GGUF embedding model."""
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    # Download model
    model_path = download_gguf_model()
    if not model_path:
        return
    
    print(f"\n🔄 Testing Qwen3-Embedding-8B-GGUF (Q4_K_M)")
    print("-" * 60)
    
    try:
        from llama_cpp import Llama
        
        # Load model with Metal GPU acceleration on macOS
        start_time = time.time()
        llm = Llama(
            model_path=model_path,
            embedding=True,  # Enable embedding mode
            n_gpu_layers=-1,  # Use Metal GPU acceleration
            verbose=False
        )
        load_time = time.time() - start_time
        print(f"✅ GGUF model loaded in {load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to load GGUF model: {e}")
        print("Note: GGUF embedding support may be limited. Falling back to alternative approach.")
        return
    
    # Load test chunks (subset for speed)
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)[:100]  # First 100 for quick test
    
    chunk_texts = [chunk["document"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    print(f"📊 Computing embeddings for {len(chunks)} chunks...")
    start_time = time.time()
    
    chunk_embeddings = []
    for i, text in enumerate(chunk_texts):
        if i % 20 == 0:
            print(f"   Processing chunk {i+1}/{len(chunk_texts)}")
        
        # Get embedding
        embedding = llm.embed(text)
        chunk_embeddings.append(embedding)
    
    chunk_embeddings = np.array(chunk_embeddings)
    embedding_time = time.time() - start_time
    print(f"✅ Embeddings computed in {embedding_time:.2f}s")
    
    # Test queries
    test_queries = [
        QueryTest(
            query="How do I create a campaign?",
            expected_chunks=["samples_openapi_yaml:createCampaigns"],
            expected_types=["endpoint"]
        ),
        QueryTest(
            query="What properties does the Campaign object have?",
            expected_chunks=["samples_openapi_yaml:Campaign"],
            expected_types=["schema"]
        ),
    ]
    
    query_times = []
    total_hits = 0
    total_possible = 0
    
    for test_query in test_queries:
        print(f"\n🔍 Query: '{test_query.query}'")
        
        # Get query embedding
        start_time = time.time()
        query_embedding = np.array([llm.embed(test_query.query)])
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Get top 5
        top_indices = similarities.argsort()[::-1][:5]
        
        hits = 0
        for i, idx in enumerate(top_indices):
            chunk_id = chunk_ids[idx]
            score = similarities[idx]
            chunk_type = chunks[idx]["metadata"].get("type", "unknown")
            
            is_expected = any(expected in chunk_id for expected in test_query.expected_chunks)
            relevance_marker = "✅" if is_expected else "❌"
            
            if is_expected and i < 3:
                hits += 1
            
            print(f"   {i+1}. {score:.3f} {relevance_marker} {chunk_id} ({chunk_type})")
        
        total_hits += hits
        total_possible += min(len(test_query.expected_chunks), 3)
    
    avg_query_time = sum(query_times) / len(query_times)
    relevance_score = total_hits / total_possible if total_possible > 0 else 0.0
    
    print(f"\n📈 Qwen3-8B-GGUF Performance Summary:")
    print(f"Load Time: {load_time:.2f}s")
    print(f"Embedding Time: {embedding_time:.2f}s ({len(chunks)} chunks)")
    print(f"Avg Query Time: {avg_query_time*1000:.1f}ms")
    print(f"Relevance Score: {relevance_score:.1%}")
    
    # Compare with 0.6B results
    print(f"\n📊 Comparison with Qwen3-0.6B:")
    print(f"{'Metric':<20} {'8B-GGUF':<15} {'0.6B Full':<15} {'Difference'}")
    print("-" * 65)
    
    print(f"{'Load Time':<20} {load_time:<14.2f}s {1.44:<14.2f}s {load_time - 1.44:+.2f}s")
    print(f"{'Query Time':<20} {avg_query_time*1000:<14.1f}ms {141.4:<14.1f}ms {avg_query_time*1000 - 141.4:+.1f}ms")
    print(f"{'Relevance':<20} {relevance_score:<14.1%} {70.0:<14.1%} {relevance_score - 0.7:+.1%}")
    
    if relevance_score > 0.7:
        improvement = (relevance_score - 0.7) / 0.7 * 100
        print(f"\n🎯 8B-GGUF shows {improvement:.1f}% improvement in relevance!")
    else:
        print(f"\n⚖️ 8B-GGUF performs similarly to 0.6B model")
    
    print(f"\n💡 GGUF vs Full Model Trade-offs:")
    print(f"   ✅ Smaller memory footprint (quantized)")
    print(f"   ✅ Faster loading than full 8B")
    print(f"   ❓ Quality vs 0.6B: {relevance_score:.1%} vs 70.0%")


def fallback_test():
    """Fallback test using sentence-transformers if GGUF doesn't work."""
    print(f"\n🔄 Fallback: Testing with sentence-transformers")
    print("(This will be slower but should work)")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Try to load a smaller quantized version if available
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        
        # Quick test
        test_text = "How do I create a campaign?"
        start_time = time.time()
        embedding = model.encode([test_text], prompt_name="query")
        query_time = time.time() - start_time
        
        print(f"✅ Fallback test successful: {query_time*1000:.1f}ms")
        print(f"   Embedding shape: {embedding.shape}")
        
    except Exception as e:
        print(f"❌ Fallback also failed: {e}")


def main():
    print("🧪 Testing Qwen3-Embedding-8B-GGUF with 4-bit Quantization")
    print("=" * 70)
    
    try:
        test_gguf_embedding()
    except Exception as e:
        print(f"❌ GGUF test failed: {e}")
        fallback_test()


if __name__ == "__main__":
    main()