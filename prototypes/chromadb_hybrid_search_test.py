#!/usr/bin/env python3
"""
ChromaDB Hybrid Search Proof of Concept

Tests ChromaDB integration with both semantic vector search and keyword/FTS search.
Validates multi-stage retrieval pipeline with reference expansion.
"""

import json
import time
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    document: str
    method: str  # "semantic", "keyword", "hybrid"


@dataclass
class QueryTestResult:
    query: str
    semantic_results: List[SearchResult]
    keyword_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    semantic_time: float
    keyword_time: float
    hybrid_time: float
    expected_chunks: List[str]


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for Qwen3-0.6B model with proper query prompts."""
    
    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        self.is_query = False  # Flag to determine if we're embedding a query
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for documents or queries."""
        if self.is_query:
            # Use query prompt for search queries
            embeddings = self.model.encode(list(input), prompt_name="query", convert_to_numpy=True)
        else:
            # No special prompt for documents
            embeddings = self.model.encode(list(input), convert_to_numpy=True)
        return embeddings.tolist()
    
    def set_query_mode(self, is_query: bool):
        """Set whether we're embedding queries (True) or documents (False)."""
        self.is_query = is_query


class ChromaDBHybridSearchTest:
    def __init__(self, chunks_file: str, sample_size: int = 200):
        self.chunks_file = chunks_file
        self.sample_size = sample_size
        
        # Setup device for embeddings
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using CUDA acceleration")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
        
        # Create custom embedding function
        print("🔄 Loading Qwen3-0.6B embedding model...")
        start_time = time.time()
        self.embedding_function = Qwen3EmbeddingFunction(device=self.device)
        load_time = time.time() - start_time
        print(f"✅ Embedding model loaded in {load_time:.2f}s")
        
        # Setup ChromaDB
        print("🔄 Setting up ChromaDB...")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Load and setup test data
        self.chunks = self._load_chunks()
        self.collection = self._setup_collection()
        
        print(f"✅ ChromaDB setup complete with {len(self.chunks)} chunks")
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load and sample chunks for testing."""
        with open(self.chunks_file, 'r') as f:
            all_chunks = json.load(f)
        
        # Deduplicate by ID (keep first occurrence)
        seen_ids = set()
        deduplicated_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                deduplicated_chunks.append(chunk)
                seen_ids.add(chunk["id"])
        
        # Take sample for faster testing
        chunks = deduplicated_chunks[:self.sample_size]
        print(f"📊 Loaded {len(chunks)} chunks (sample from {len(deduplicated_chunks)} total, {len(all_chunks) - len(deduplicated_chunks)} duplicates removed)")
        return chunks
    
    def _setup_collection(self) -> chromadb.Collection:
        """Create ChromaDB collection and index chunks."""
        collection_name = "api_knowledge_test"
        
        # Reset collection if exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection with custom Qwen3 embedding function
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Prepare data for indexing
        print("🔄 Computing embeddings for chunk indexing...")
        start_time = time.time()
        
        chunk_ids = [chunk["id"] for chunk in self.chunks]
        chunk_documents = [chunk["document"] for chunk in self.chunks]
        
        # Convert metadata to ChromaDB-compatible format (no lists allowed)
        chunk_metadatas = []
        for chunk in self.chunks:
            metadata = {}
            for key, value in chunk["metadata"].items():
                # Convert any list to comma-separated string
                if isinstance(value, list):
                    metadata[key] = ",".join(str(v) for v in value) if value else ""
                else:
                    metadata[key] = value
            chunk_metadatas.append(metadata)
        
        # Index chunks in ChromaDB with Qwen3 embeddings
        print("🔄 Indexing chunks in ChromaDB with Qwen3 embeddings...")
        
        # Set embedding function to document mode
        self.embedding_function.set_query_mode(False)
        
        start_time = time.time()
        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadatas
        )
        index_time = time.time() - start_time
        
        print(f"✅ Indexed {len(chunk_ids)} chunks in {index_time:.2f}s")
        return collection
    
    def semantic_search(self, query: str, n_results: int = 5) -> Tuple[List[SearchResult], float]:
        """Perform pure semantic vector search using Qwen3 embeddings."""
        start_time = time.time()
        
        # Set embedding function to query mode for proper prompt handling
        self.embedding_function.set_query_mode(True)
        
        # Search in ChromaDB using Qwen3 embeddings
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Reset to document mode
        self.embedding_function.set_query_mode(False)
        
        search_time = time.time() - start_time
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                chunk_id=results['ids'][0][i],
                score=results['distances'][0][i],  # ChromaDB returns distances (lower = better)
                chunk_type=results['metadatas'][0][i].get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",  # Truncate for display
                method="semantic"
            ))
        
        return search_results, search_time
    
    def keyword_search(self, query: str, n_results: int = 5) -> Tuple[List[SearchResult], float]:
        """Perform pure keyword/FTS search using ChromaDB's where_document."""
        start_time = time.time()
        
        # Build keyword filters
        query_words = query.lower().split()
        
        # Create keyword search filter
        keyword_conditions = []
        
        # Add contains filters for each significant word
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                keyword_conditions.append({"$contains": word})
        
        # Combine with OR if multiple keywords
        if len(keyword_conditions) > 1:
            where_document = {"$or": keyword_conditions}
        elif len(keyword_conditions) == 1:
            where_document = keyword_conditions[0]
        else:
            where_document = {"$contains": query.lower()}
        
        # Set embedding function to query mode
        self.embedding_function.set_query_mode(True)
        
        # Search with keyword filtering using ChromaDB's FTS
        results = self.collection.query(
            query_texts=[query],
            where_document=where_document,
            n_results=n_results
        )
        
        # Reset to document mode
        self.embedding_function.set_query_mode(False)
        
        search_time = time.time() - start_time
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                chunk_id=results['ids'][0][i],
                score=results['distances'][0][i],
                chunk_type=results['metadatas'][0][i].get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",
                method="keyword"
            ))
        
        return search_results, search_time
    
    def hybrid_search(self, query: str, n_results: int = 5) -> Tuple[List[SearchResult], float]:
        """Perform hybrid search combining semantic and keyword approaches."""
        start_time = time.time()
        
        # Get both semantic and keyword results
        semantic_results, _ = self.semantic_search(query, n_results)
        keyword_results, _ = self.keyword_search(query, n_results)
        
        # Combine and deduplicate by chunk_id
        combined_results = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            combined_results[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                score=result.score * 0.7,  # Weight semantic slightly higher
                chunk_type=result.chunk_type,
                document=result.document,
                method="hybrid"
            )
        
        # Add keyword results, boosting if already present
        for result in keyword_results:
            if result.chunk_id in combined_results:
                # Boost existing result
                combined_results[result.chunk_id].score = (
                    combined_results[result.chunk_id].score * 0.5 + result.score * 0.3
                )
            else:
                # Add new keyword result
                combined_results[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    score=result.score * 0.5,  # Lower weight for keyword-only
                    chunk_type=result.chunk_type,
                    document=result.document,
                    method="hybrid"
                )
        
        # Sort by score and take top n_results
        sorted_results = sorted(combined_results.values(), key=lambda x: x.score)[:n_results]
        
        search_time = time.time() - start_time
        return sorted_results, search_time
    
    def test_reference_expansion(self, primary_results: List[SearchResult]) -> List[SearchResult]:
        """Test multi-stage retrieval with reference expansion."""
        print(f"\n🔗 Testing reference expansion...")
        
        # Extract all ref_ids from primary results
        all_ref_ids = set()
        for result in primary_results:
            # Find the full chunk data to get metadata
            chunk_data = next((c for c in self.chunks if c["id"] == result.chunk_id), None)
            if chunk_data and "ref_ids" in chunk_data["metadata"]:
                # Handle both list and string formats
                ref_ids = chunk_data["metadata"]["ref_ids"]
                if isinstance(ref_ids, list):
                    all_ref_ids.update(ref_ids)
                elif isinstance(ref_ids, str) and ref_ids:
                    all_ref_ids.update(ref_ids.split(","))
        
        if not all_ref_ids:
            print("   No references found in primary results")
            return []
        
        print(f"   Found {len(all_ref_ids)} referenced chunks to retrieve")
        
        # Direct lookup by IDs
        start_time = time.time()
        try:
            ref_results = self.collection.get(ids=list(all_ref_ids))
            lookup_time = time.time() - start_time
            
            # Convert to SearchResult objects
            expanded_results = []
            for i in range(len(ref_results['ids'])):
                expanded_results.append(SearchResult(
                    chunk_id=ref_results['ids'][i],
                    score=0.5,  # Neutral score for referenced chunks
                    chunk_type=ref_results['metadatas'][i].get('type', 'unknown'),
                    document=ref_results['documents'][i][:100] + "...",
                    method="reference_expansion"
                ))
            
            print(f"   ✅ Retrieved {len(expanded_results)} referenced chunks in {lookup_time*1000:.1f}ms")
            return expanded_results
            
        except Exception as e:
            print(f"   ❌ Reference expansion failed: {e}")
            return []
    
    def run_query_tests(self) -> List[QueryTestResult]:
        """Run comprehensive tests comparing all search methods."""
        test_queries = [
            {
                "query": "How do I create a campaign?",
                "expected": ["samples_openapi_yaml:createCampaigns", "samples_openapi_json:CreateSponsoredBrandsCampaigns"]
            },
            {
                "query": "What properties does the Campaign object have?",
                "expected": ["samples_openapi_yaml:Campaign", "samples_openapi_json:Campaign"]
            },
            {
                "query": "createCampaign endpoint",
                "expected": ["samples_openapi_yaml:createCampaigns"]
            },
            {
                "query": "ACCESS_DENIED error",
                "expected": ["samples_openapi_json:SBTargetingAccessDeniedExceptionResponseContent"]
            },
            {
                "query": "campaign update",
                "expected": ["samples_openapi_yaml:updateCampaigns"]
            }
        ]
        
        print(f"\n🧪 Running Query Tests")
        print("=" * 70)
        
        results = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            # Test semantic search
            semantic_results, semantic_time = self.semantic_search(query)
            print(f"   Semantic ({semantic_time*1000:.1f}ms):")
            for i, result in enumerate(semantic_results[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
            
            # Test keyword search
            try:
                keyword_results, keyword_time = self.keyword_search(query)
                print(f"   Keyword ({keyword_time*1000:.1f}ms):")
                for i, result in enumerate(keyword_results[:3]):
                    relevance = "✅" if result.chunk_id in expected else "❌"
                    print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
            except Exception as e:
                print(f"   Keyword search failed: {e}")
                keyword_results, keyword_time = [], 0
            
            # Test hybrid search
            hybrid_results, hybrid_time = self.hybrid_search(query)
            print(f"   Hybrid ({hybrid_time*1000:.1f}ms):")
            for i, result in enumerate(hybrid_results[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
            
            # Test reference expansion on best results
            if hybrid_results:
                expanded = self.test_reference_expansion(hybrid_results[:2])
                if expanded:
                    print(f"   Reference Expansion:")
                    for i, result in enumerate(expanded[:3]):
                        print(f"      +{i+1}. {result.chunk_id} ({result.chunk_type})")
            
            results.append(QueryTestResult(
                query=query,
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                hybrid_results=hybrid_results,
                semantic_time=semantic_time,
                keyword_time=keyword_time,
                hybrid_time=hybrid_time,
                expected_chunks=expected
            ))
        
        return results
    
    def calculate_relevance_score(self, results: List[SearchResult], expected: List[str]) -> float:
        """Calculate relevance score for top-3 results."""
        top_3_ids = [r.chunk_id for r in results[:3]]
        hits = sum(1 for chunk_id in top_3_ids if chunk_id in expected)
        return hits / min(len(expected), 3) if expected else 0.0
    
    def print_summary(self, test_results: List[QueryTestResult]):
        """Print comprehensive test summary."""
        print(f"\n📈 ChromaDB Hybrid Search Test Summary")
        print("=" * 70)
        
        # Calculate average metrics
        semantic_relevance = []
        keyword_relevance = []
        hybrid_relevance = []
        semantic_times = []
        keyword_times = []
        hybrid_times = []
        
        for result in test_results:
            semantic_relevance.append(self.calculate_relevance_score(result.semantic_results, result.expected_chunks))
            keyword_relevance.append(self.calculate_relevance_score(result.keyword_results, result.expected_chunks))
            hybrid_relevance.append(self.calculate_relevance_score(result.hybrid_results, result.expected_chunks))
            semantic_times.append(result.semantic_time * 1000)
            keyword_times.append(result.keyword_time * 1000)
            hybrid_times.append(result.hybrid_time * 1000)
        
        avg_semantic_relevance = sum(semantic_relevance) / len(semantic_relevance)
        avg_keyword_relevance = sum(keyword_relevance) / len(keyword_relevance)
        avg_hybrid_relevance = sum(hybrid_relevance) / len(hybrid_relevance)
        avg_semantic_time = sum(semantic_times) / len(semantic_times)
        avg_keyword_time = sum(keyword_times) / len(keyword_times)
        avg_hybrid_time = sum(hybrid_times) / len(hybrid_times)
        
        print(f"{'Method':<15} {'Relevance':<12} {'Avg Time':<12} {'Performance'}") 
        print("-" * 60)
        print(f"{'Semantic':<15} {avg_semantic_relevance:<11.1%} {avg_semantic_time:<11.1f}ms {'baseline'}")
        print(f"{'Keyword':<15} {avg_keyword_relevance:<11.1%} {avg_keyword_time:<11.1f}ms {'+' if avg_keyword_relevance > avg_semantic_relevance else '-'}{abs(avg_keyword_relevance - avg_semantic_relevance):.1%}")
        print(f"{'Hybrid':<15} {avg_hybrid_relevance:<11.1%} {avg_hybrid_time:<11.1f}ms {'+' if avg_hybrid_relevance > avg_semantic_relevance else '-'}{abs(avg_hybrid_relevance - avg_semantic_relevance):.1%}")
        
        # Determine best approach
        best_method = "semantic"
        best_score = avg_semantic_relevance
        
        if avg_keyword_relevance > best_score:
            best_method = "keyword"
            best_score = avg_keyword_relevance
        
        if avg_hybrid_relevance > best_score:
            best_method = "hybrid"
            best_score = avg_hybrid_relevance
        
        print(f"\n🏆 Best Method: {best_method.title()} ({best_score:.1%} relevance)")
        
        # Performance analysis
        print(f"\n💡 Key Findings:")
        if avg_hybrid_relevance > avg_semantic_relevance:
            improvement = (avg_hybrid_relevance - avg_semantic_relevance) / avg_semantic_relevance * 100
            print(f"   ✅ Hybrid improves relevance by {improvement:.1f}% over semantic-only")
        else:
            print(f"   ⚠️ Hybrid doesn't significantly improve over semantic search")
        
        if avg_keyword_time < avg_semantic_time:
            speedup = avg_semantic_time / avg_keyword_time
            print(f"   ⚡ Keyword search is {speedup:.1f}x faster than semantic")
        
        if avg_hybrid_relevance >= 0.7:
            print(f"   🎯 Hybrid search meets 70% relevance target")
        else:
            print(f"   ❌ Hybrid search below 70% target ({avg_hybrid_relevance:.1%})")


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run the chunker first: python prototypes/prototype_chunker.py")
        return
    
    print("🚀 ChromaDB Hybrid Search Proof of Concept")
    print("=" * 60)
    
    # Run test with sample size
    tester = ChromaDBHybridSearchTest(chunks_file, sample_size=200)
    test_results = tester.run_query_tests()
    tester.print_summary(test_results)
    
    print(f"\n✅ Test complete!")
    print(f"Tested {len(test_results)} queries against {len(tester.chunks)} chunks")


if __name__ == "__main__":
    main()