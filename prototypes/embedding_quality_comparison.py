#!/usr/bin/env python3
"""
Comprehensive embedding model comparison with quality scoring.
Tests Qwen3, all-MiniLM-L6-v2, and GTE-large-en-v1.5 against manual relevance scores.
"""

import json
import time
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ModelResult:
    model_name: str
    load_time: float
    embedding_time: float
    avg_query_time: float
    results: List[Dict[str, Any]]
    relevance_score: float  # Manual evaluation score


@dataclass
class QueryTest:
    query: str
    expected_chunks: List[str]  # Manually curated relevant chunk IDs
    expected_types: List[str]   # Expected chunk types (endpoint/schema)


class EmbeddingQualityComparison:
    def __init__(self, chunks_file: str):
        self.chunks = self._load_chunks(chunks_file)
        self.chunk_texts = [chunk["document"] for chunk in self.chunks]
        self.chunk_ids = [chunk["id"] for chunk in self.chunks]
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using CUDA acceleration")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
        
        # Define test queries with expected results (manual curation)
        self.test_queries = [
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
        
        print(f"📊 Loaded {len(self.chunks)} chunks for comparison")
    
    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(chunks_file, 'r') as f:
            return json.load(f)
    
    def test_model(self, model_name: str, special_config: Dict = None) -> ModelResult:
        """Test a specific embedding model."""
        print(f"\n🔄 Testing {model_name}")
        print("-" * 60)
        
        # Load model
        start_time = time.time()
        try:
            if special_config:
                model = SentenceTransformer(model_name, device=self.device, **special_config)
            else:
                model = SentenceTransformer(model_name, device=self.device)
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
        
        # Compute embeddings for all chunks
        print("🔄 Computing embeddings for all chunks...")
        start_time = time.time()
        chunk_embeddings = model.encode(self.chunk_texts, show_progress_bar=True)
        embedding_time = time.time() - start_time
        print(f"✅ Embeddings computed in {embedding_time:.2f}s")
        
        # Test each query
        query_times = []
        results = []
        
        for test_query in self.test_queries:
            print(f"\n🔍 Query: '{test_query.query}'")
            
            # Encode query
            start_time = time.time()
            if "Qwen3-Embedding" in model_name:
                # Use special prompt for Qwen3
                query_embedding = model.encode([test_query.query], prompt_name="query")
            else:
                query_embedding = model.encode([test_query.query])
            
            # Compute similarities
            if "Qwen3-Embedding" in model_name and hasattr(model, 'similarity'):
                similarities = model.similarity(query_embedding, chunk_embeddings).numpy()[0]
            else:
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Get top 5 results
            top_indices = similarities.argsort()[::-1][:5]
            top_results = []
            
            for i, idx in enumerate(top_indices):
                chunk_id = self.chunk_ids[idx]
                score = similarities[idx]
                chunk_type = self.chunks[idx]["metadata"].get("type", "unknown")
                
                top_results.append({
                    "rank": i + 1,
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "type": chunk_type
                })
                
                # Show relevance
                is_expected = chunk_id in test_query.expected_chunks
                relevance_marker = "✅" if is_expected else "❌"
                print(f"   {i+1}. {score:.3f} {relevance_marker} {chunk_id} ({chunk_type})")
            
            results.append({
                "query": test_query.query,
                "expected": test_query.expected_chunks,
                "results": top_results,
                "query_time": query_time
            })
        
        avg_query_time = np.mean(query_times)
        
        # Calculate overall relevance score
        relevance_score = self._calculate_relevance_score(results)
        
        print(f"\n📈 Model Performance Summary:")
        print(f"Load Time: {load_time:.2f}s")
        print(f"Embedding Time: {embedding_time:.2f}s") 
        print(f"Avg Query Time: {avg_query_time*1000:.1f}ms")
        print(f"Relevance Score: {relevance_score:.1%}")
        
        return ModelResult(
            model_name=model_name,
            load_time=load_time,
            embedding_time=embedding_time,
            avg_query_time=avg_query_time,
            results=results,
            relevance_score=relevance_score
        )
    
    def _calculate_relevance_score(self, results: List[Dict]) -> float:
        """Calculate relevance score based on expected vs actual results."""
        total_score = 0
        total_queries = len(results)
        
        for result in results:
            expected_chunks = set(result["expected"])
            found_chunks = {r["chunk_id"] for r in result["results"][:3]}  # Top 3
            
            # Score based on intersection
            intersection = len(expected_chunks & found_chunks)
            max_possible = len(expected_chunks)
            
            if max_possible > 0:
                query_score = intersection / max_possible
            else:
                query_score = 1.0 if len(found_chunks) > 0 else 0.0
            
            total_score += query_score
        
        return total_score / total_queries if total_queries > 0 else 0.0
    
    def run_comparison(self) -> List[ModelResult]:
        """Run comparison across all models."""
        print("🧪 Embedding Model Quality Comparison")
        print("=" * 80)
        
        models_to_test = [
            {
                "name": "all-MiniLM-L6-v2",
                "config": None
            },
            {
                "name": "Qwen/Qwen3-Embedding-0.6B", 
                "config": None
            },
            {
                "name": "Alibaba-NLP/gte-large-en-v1.5",
                "config": {"trust_remote_code": True}
            }
        ]
        
        results = []
        
        for model_config in models_to_test:
            try:
                result = self.test_model(model_config["name"], model_config["config"])
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Failed to test {model_config['name']}: {e}")
                continue
        
        # Print final comparison
        print(f"\n🏆 Final Comparison")
        print("=" * 80)
        print(f"{'Model':<35} {'Relevance':<12} {'Query Time':<12} {'Load Time':<12}")
        print("-" * 80)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for result in results:
            print(f"{result.model_name:<35} {result.relevance_score:<11.1%} {result.avg_query_time*1000:<11.1f}ms {result.load_time:<11.2f}s")
        
        if results:
            best_model = results[0]
            print(f"\n🥇 Best Overall: {best_model.model_name}")
            print(f"   Relevance: {best_model.relevance_score:.1%}")
            print(f"   Speed: {best_model.avg_query_time*1000:.1f}ms per query")
        
        return results


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run the chunker first: python prototypes/prototype_chunker.py")
        return
    
    comparator = EmbeddingQualityComparison(chunks_file)
    results = comparator.run_comparison()
    
    print(f"\n✅ Comparison complete!")
    if results:
        print(f"Tested {len(results)} models successfully")


if __name__ == "__main__":
    main()