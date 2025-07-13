#!/usr/bin/env python3
"""
Embedding Model Validation - Test Qwen3-Embedding-0.6B vs keyword simulation.
Compares real embeddings against our current keyword matching for chunk retrieval.
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
class RetrievalResult:
    query: str
    keyword_chunks: List[str]  # chunk IDs from keyword matching
    embedding_chunks: List[str]  # chunk IDs from embedding similarity
    overlap_score: float  # how much they agree
    keyword_time: float
    embedding_time: float


class EmbeddingValidator:
    def __init__(self, chunks_file: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        # Check for MPS (Apple Silicon) support
        if torch.backends.mps.is_available():
            device = "mps"
            print("🍎 Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using CUDA acceleration")
        else:
            device = "cpu"
            print("💻 Using CPU (no acceleration)")
        
        print(f"🔄 Loading embedding model: {model_name}")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s on {device}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            print("🔄 Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            load_time = time.time() - start_time
            print(f"✅ Fallback model loaded in {load_time:.2f}s on {device}")
        
        self.chunks = self._load_chunks(chunks_file)
        self.chunk_texts = [chunk["document"] for chunk in self.chunks]
        self.chunk_ids = [chunk["id"] for chunk in self.chunks]
        
        print(f"📊 Loaded {len(self.chunks)} chunks")
        print("🔄 Computing embeddings for all chunks...")
        
        start_time = time.time()
        # For Qwen3, documents don't need special prompts (only queries do)
        self.chunk_embeddings = self.model.encode(self.chunk_texts, show_progress_bar=True)
        embedding_time = time.time() - start_time
        print(f"✅ Embeddings computed in {embedding_time:.2f}s")
    
    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(chunks_file, 'r') as f:
            return json.load(f)
    
    def keyword_search_simulation(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Simulate keyword search like our current analyzer."""
        start_time = time.time()
        
        query_words = set(query.lower().split())
        
        # Add common variations and synonyms (from chunk_analyzer.py)
        if "campaign" in query_words:
            query_words.update(["campaigns", "campaignid"])
        if "create" in query_words:
            query_words.update(["post", "add", "new"])
        if "get" in query_words or "list" in query_words:
            query_words.update(["retrieve", "fetch", "show"])
        if "error" in query_words:
            query_words.update(["exception", "failure", "problem"])
        if "field" in query_words:
            query_words.update(["property", "attribute", "parameter"])
        
        scored_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            score = 0
            content = (chunk["document"] + " " + str(chunk["metadata"])).lower()
            
            # Keyword matching with different weights
            for word in query_words:
                if word in content:
                    # Higher score for exact matches in operation names
                    if word in chunk["metadata"].get("operationId", "").lower():
                        score += 3
                    # Medium score for matches in document content
                    elif word in chunk["document"].lower():
                        score += 2
                    # Lower score for matches in metadata
                    else:
                        score += 1
            
            # Boost score for endpoint chunks vs schema chunks based on query
            if any(verb in query_words for verb in ["create", "get", "list", "update", "delete"]):
                if chunk["metadata"].get("type") == "endpoint":
                    score += 1
            elif any(term in query_words for term in ["field", "property", "schema", "object"]):
                if chunk["metadata"].get("type") == "schema":
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, i))
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunk_ids = [self.chunk_ids[i] for score, i in scored_chunks[:top_k]]
        
        search_time = time.time() - start_time
        return top_chunk_ids, search_time
    
    def embedding_search(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Perform semantic search using embeddings."""
        start_time = time.time()
        
        # Check if this is Qwen3 model and use proper prompt
        if "Qwen3-Embedding" in str(self.model):
            # Use query prompt for Qwen3
            query_embedding = self.model.encode([query], prompt_name="query")
        else:
            # Standard encoding for other models
            query_embedding = self.model.encode([query])
        
        # Compute similarities using model's similarity method if available
        if hasattr(self.model, 'similarity'):
            similarities = self.model.similarity(query_embedding, self.chunk_embeddings).numpy()[0]
        else:
            similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunk_ids = [self.chunk_ids[i] for i in top_indices]
        
        search_time = time.time() - start_time
        return top_chunk_ids, search_time
    
    def calculate_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate overlap between two lists of chunk IDs."""
        set1, set2 = set(list1), set(list2)
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union  # Jaccard similarity
    
    def validate_query(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Compare keyword vs embedding retrieval for a single query."""
        print(f"\n🔍 Testing: \"{query}\"")
        
        # Keyword search
        keyword_chunks, keyword_time = self.keyword_search_simulation(query, top_k)
        print(f"   Keyword ({keyword_time*1000:.1f}ms): {len(keyword_chunks)} chunks")
        for chunk_id in keyword_chunks[:2]:
            print(f"      - {chunk_id}")
        
        # Embedding search  
        embedding_chunks, embedding_time = self.embedding_search(query, top_k)
        print(f"   Embedding ({embedding_time*1000:.1f}ms): {len(embedding_chunks)} chunks")
        for chunk_id in embedding_chunks[:2]:
            print(f"      - {chunk_id}")
        
        # Calculate overlap
        overlap = self.calculate_overlap(keyword_chunks, embedding_chunks)
        print(f"   Overlap: {overlap:.2%}")
        
        return RetrievalResult(
            query=query,
            keyword_chunks=keyword_chunks,
            embedding_chunks=embedding_chunks,
            overlap_score=overlap,
            keyword_time=keyword_time,
            embedding_time=embedding_time
        )
    
    def run_validation_suite(self) -> Dict[str, Any]:
        """Run comprehensive validation comparing both approaches."""
        test_queries = [
            # Endpoint discovery queries
            "How do I create a new sponsored display campaign?",
            "How do I get a list of campaigns?", 
            "How to update a campaign?",
            "How do I delete an ad group?",
            
            # Schema inspection queries
            "What fields are required in CreateCampaignRequest?",
            "What properties does the Campaign object have?",
            "What fields are in UpdateAttributesRequestContent?",
            
            # Error handling queries
            "What causes ACCESS_DENIED error in AdCatalog API?",
            "What are the possible error responses for campaign creation?",
            "What does error code 400 mean?",
            
            # Cross-API queries  
            "Which APIs support campaign frequency capping?",
            "What APIs support budget rules?",
            
            # Field-specific queries
            "What are valid values for campaign state?",
            "What is the format for targeting criteria?",
            "What are the required fields for creating an ad?",
        ]
        
        print("🧪 Running Embedding Validation Suite")
        print("=" * 60)
        
        results = []
        total_overlap = 0
        total_keyword_time = 0
        total_embedding_time = 0
        
        for query in test_queries:
            result = self.validate_query(query)
            results.append(result)
            total_overlap += result.overlap_score
            total_keyword_time += result.keyword_time
            total_embedding_time += result.embedding_time
        
        avg_overlap = total_overlap / len(test_queries)
        avg_keyword_time = total_keyword_time / len(test_queries)
        avg_embedding_time = total_embedding_time / len(test_queries)
        
        print(f"\n📈 Validation Results")
        print("=" * 60)
        print(f"Average Overlap: {avg_overlap:.2%}")
        print(f"Average Keyword Time: {avg_keyword_time*1000:.1f}ms")
        print(f"Average Embedding Time: {avg_embedding_time*1000:.1f}ms")
        
        # Categorize overlaps
        high_overlap = [r for r in results if r.overlap_score >= 0.6]
        medium_overlap = [r for r in results if 0.3 <= r.overlap_score < 0.6]
        low_overlap = [r for r in results if r.overlap_score < 0.3]
        
        print(f"High Overlap (≥60%): {len(high_overlap)}")
        print(f"Medium Overlap (30-59%): {len(medium_overlap)}")  
        print(f"Low Overlap (<30%): {len(low_overlap)}")
        
        if low_overlap:
            print(f"\n❌ Queries with low keyword-embedding agreement:")
            for result in low_overlap:
                print(f"   {result.overlap_score:.1%} - {result.query}")
        
        # Performance comparison
        speed_ratio = avg_embedding_time / avg_keyword_time
        print(f"\nPerformance:")
        print(f"Embedding is {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'} than keyword")
        
        return {
            "average_overlap": avg_overlap,
            "results": results,
            "high_overlap": len(high_overlap),
            "medium_overlap": len(medium_overlap),
            "low_overlap": len(low_overlap),
            "avg_keyword_time": avg_keyword_time,
            "avg_embedding_time": avg_embedding_time,
            "speed_ratio": speed_ratio
        }


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run the chunker first: python prototypes/prototype_chunker.py")
        return
    
    # Test Qwen3 embedding model
    print("🚀 Testing Qwen3-Embedding-0.6B vs Keyword Simulation")
    print("=" * 60)
    
    validator = EmbeddingValidator(chunks_file, "Qwen/Qwen3-Embedding-0.6B")
    summary = validator.run_validation_suite()
    
    print(f"\n✅ Validation complete!")
    print(f"Keyword-Embedding Agreement: {summary['average_overlap']:.1%}")
    
    if summary['average_overlap'] >= 0.6:
        print("🎉 High agreement! Embeddings can replace keyword simulation")
    elif summary['average_overlap'] >= 0.4:
        print("⚠️  Moderate agreement. Consider hybrid approach")
    else:
        print("❌ Low agreement. Keyword simulation may be better")
    
    print(f"Performance: Embedding {summary['speed_ratio']:.1f}x {'slower' if summary['speed_ratio'] > 1 else 'faster'}")


if __name__ == "__main__":
    main()