#!/usr/bin/env python3
"""
Quick test of Qwen3-Embedding-8B to see if larger model performs significantly better
"""

import json
import time
import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QueryTest:
    query: str
    expected_chunks: List[str]  # Manually curated relevant chunk IDs
    expected_types: List[str]   # Expected chunk types (endpoint/schema)


def test_qwen3_8b():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA acceleration")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print(f"\n🔄 Testing Qwen/Qwen3-Embedding-8B")
    print("-" * 60)
    
    # Load model
    start_time = time.time()
    try:
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load Qwen3-8B: {e}")
        return
    
    # Load chunks (use subset for faster testing)
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)[:200]  # First 200 chunks for speed
    
    chunk_texts = [chunk["document"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    print(f"📊 Computing embeddings for {len(chunks)} chunks...")
    start_time = time.time()
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embedding_time = time.time() - start_time
    print(f"✅ Embeddings computed in {embedding_time:.2f}s")
    
    # Test queries with expected results (same as quality comparison)
    test_queries = [
        QueryTest(
            query="How do I create a campaign?",
            expected_chunks=["samples_openapi_yaml:createCampaigns", "samples_openapi_json:CreateSponsoredBrandsCampaigns"],
            expected_types=["endpoint"]
        ),
        QueryTest(
            query="What properties does the Campaign object have?",
            expected_chunks=["samples_openapi_yaml:Campaign", "samples_openapi_json:Campaign"],
            expected_types=["schema"]
        ),
        QueryTest(
            query="How to update a campaign?",
            expected_chunks=["samples_openapi_yaml:updateCampaigns"],
            expected_types=["endpoint"]
        ),
        QueryTest(
            query="What fields are required in CreateCampaignRequest?",
            expected_chunks=["samples_openapi_yaml:CreateCampaign", "samples_openapi_yaml:CreateCampaignRequest"],
            expected_types=["schema", "endpoint"]
        ),
        QueryTest(
            query="What causes ACCESS_DENIED error?",
            expected_chunks=["samples_openapi_json:SBTargetingAccessDeniedExceptionResponseContent"],
            expected_types=["schema"]
        ),
    ]
    
    query_times = []
    total_hits = 0
    total_possible = 0
    
    for test_query in test_queries:
        print(f"\n🔍 Query: '{test_query.query}'")
        
        # Encode query with prompt
        start_time = time.time()
        query_embedding = model.encode([test_query.query], prompt_name="query")
        
        # Compute similarities
        similarities = model.similarity(query_embedding, chunk_embeddings).numpy()[0]
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Get top 5 results
        top_indices = similarities.argsort()[::-1][:5]
        
        hits = 0
        for i, idx in enumerate(top_indices):
            chunk_id = chunk_ids[idx]
            score = similarities[idx]
            chunk_type = chunks[idx]["metadata"].get("type", "unknown")
            
            # Check if this is an expected result
            is_expected = any(expected in chunk_id for expected in test_query.expected_chunks)
            relevance_marker = "✅" if is_expected else "❌"
            
            if is_expected and i < 3:  # Top 3 count as hits
                hits += 1
            
            print(f"   {i+1}. {score:.3f} {relevance_marker} {chunk_id} ({chunk_type})")
        
        total_hits += hits
        total_possible += min(len(test_query.expected_chunks), 3)  # Max 3 possible hits per query
    
    avg_query_time = sum(query_times) / len(query_times)
    relevance_score = total_hits / total_possible if total_possible > 0 else 0.0
    
    print(f"\n📈 Qwen3-8B Performance Summary:")
    print(f"Load Time: {load_time:.2f}s")
    print(f"Embedding Time: {embedding_time:.2f}s ({len(chunks)} chunks)")
    print(f"Avg Query Time: {avg_query_time*1000:.1f}ms")
    print(f"Relevance Score: {relevance_score:.1%}")
    
    # Compare with 0.6B results
    print(f"\n📊 Comparison with Qwen3-0.6B:")
    print(f"{'Metric':<20} {'8B Model':<15} {'0.6B Model':<15} {'Difference'}")
    print("-" * 65)
    
    # Extrapolate 8B timing to full dataset
    estimated_full_embedding = embedding_time * (2079 / len(chunks))
    
    print(f"{'Load Time':<20} {load_time:<14.2f}s {1.44:<14.2f}s {load_time - 1.44:+.2f}s")
    print(f"{'Embedding Time':<20} {estimated_full_embedding:<14.1f}s {20.3:<14.1f}s {estimated_full_embedding - 20.3:+.1f}s")
    print(f"{'Query Time':<20} {avg_query_time*1000:<14.1f}ms {141.4:<14.1f}ms {avg_query_time*1000 - 141.4:+.1f}ms")
    print(f"{'Relevance':<20} {relevance_score:<14.1%} {70.0:<14.1%} {relevance_score - 0.7:+.1%}")
    
    if relevance_score > 0.7:
        improvement = (relevance_score - 0.7) / 0.7 * 100
        print(f"\n🎯 8B model shows {improvement:.1f}% improvement in relevance!")
    elif relevance_score == 0.7:
        print(f"\n⚖️  8B model performs similarly to 0.6B model")
    else:
        print(f"\n⚠️  8B model underperforms compared to 0.6B model")
    
    print(f"\n💡 Recommendation:")
    if relevance_score > 0.75:
        print("   8B model significantly better - worth the extra resources")
    elif relevance_score > 0.7:
        print("   8B model marginally better - consider resource trade-offs")
    else:
        print("   Stick with 0.6B model - 8B doesn't justify the cost")


if __name__ == "__main__":
    test_qwen3_8b()