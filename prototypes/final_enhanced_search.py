#!/usr/bin/env python3
"""
Final Enhanced Search Implementation

Combines:
1. Pre-processed chunks with LLM-generated metadata
2. Qwen3 embeddings on enriched content
3. ChromaDB with metadata filtering
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
class EnhancedSearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    document: str
    method: str
    llm_keywords: Optional[List[str]] = None
    matched_on: Optional[str] = None  # original, keywords, semantic


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for Qwen3-0.6B model."""
    
    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        self.is_query = False
    
    def __call__(self, input: Documents) -> Embeddings:
        if self.is_query:
            embeddings = self.model.encode(list(input), prompt_name="query", convert_to_numpy=True)
        else:
            embeddings = self.model.encode(list(input), convert_to_numpy=True)
        return embeddings.tolist()
    
    def set_query_mode(self, is_query: bool):
        self.is_query = is_query


class FinalEnhancedSearch:
    """Search implementation using pre-enhanced chunks."""
    
    def __init__(self, chunks_file: str, enhanced_chunks_file: Optional[str] = None):
        self.chunks_file = chunks_file
        self.enhanced_chunks_file = enhanced_chunks_file
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using MPS for embeddings")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using CUDA for embeddings")
        else:
            self.device = "cpu"
            print("💻 Using CPU for embeddings")
        
        # Initialize embedding model
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
        
        # Load and process chunks
        self.chunks = self._load_and_merge_chunks()
        self.collection = self._setup_enhanced_collection()
        
        print(f"✅ Enhanced search ready with {len(self.chunks)} chunks")
    
    def _load_and_merge_chunks(self) -> List[Dict[str, Any]]:
        """Load original chunks and merge with LLM enhancements if available."""
        
        # Load original chunks
        with open(self.chunks_file, 'r') as f:
            original_chunks = json.load(f)
        
        # Create lookup dict
        chunks_dict = {chunk['id']: chunk for chunk in original_chunks}
        
        # Merge with enhanced chunks if available
        if self.enhanced_chunks_file and Path(self.enhanced_chunks_file).exists():
            print("🔄 Loading LLM-enhanced metadata...")
            with open(self.enhanced_chunks_file, 'r') as f:
                enhanced_chunks = json.load(f)
            
            # Merge enhancements
            enhanced_count = 0
            for enhanced in enhanced_chunks:
                chunk_id = enhanced['id']
                if chunk_id in chunks_dict and 'llm_metadata' in enhanced:
                    chunks_dict[chunk_id]['llm_metadata'] = enhanced['llm_metadata']
                    enhanced_count += 1
            
            print(f"✅ Merged {enhanced_count} LLM enhancements")
        
        # Sample for testing
        chunks = list(chunks_dict.values())[:200]
        
        # Stats
        enhanced = sum(1 for c in chunks if 'llm_metadata' in c)
        print(f"📊 Total chunks: {len(chunks)} ({enhanced} with LLM metadata)")
        
        return chunks
    
    def _create_enriched_document(self, chunk: Dict[str, Any]) -> str:
        """Create enriched document content including LLM metadata."""
        
        # Start with original document
        enriched = chunk['document']
        
        # Add LLM-generated content if available
        if 'llm_metadata' in chunk:
            meta = chunk['llm_metadata']
            
            # Add search keywords
            if 'search_keywords' in meta and meta['search_keywords']:
                enriched += "\n\nSearch Keywords: " + ", ".join(meta['search_keywords'])
            
            # Add semantic context
            if 'semantic_context' in meta and meta['semantic_context']:
                enriched += "\n\nContext: " + " ".join(meta['semantic_context'])
            
            # Add usage patterns
            if 'usage_patterns' in meta and meta['usage_patterns']:
                enriched += "\n\nUsage: " + " ".join(meta['usage_patterns'])
            
            # Add alternative queries
            if 'alternative_queries' in meta and meta['alternative_queries']:
                enriched += "\n\nAlso searched as: " + " ".join(meta['alternative_queries'])
        
        return enriched
    
    def _setup_enhanced_collection(self) -> chromadb.Collection:
        """Create ChromaDB collection with enriched documents."""
        collection_name = "api_knowledge_final"
        
        # Reset collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("🔄 Indexing enriched chunks...")
        start_time = time.time()
        
        chunk_ids = []
        chunk_documents = []
        chunk_metadatas = []
        
        for chunk in self.chunks:
            chunk_ids.append(chunk['id'])
            
            # Create enriched document
            enriched_doc = self._create_enriched_document(chunk)
            chunk_documents.append(enriched_doc)
            
            # Prepare metadata
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, list):
                    metadata[key] = ",".join(str(v) for v in value) if value else ""
                else:
                    metadata[key] = value
            
            # Add flags for LLM enhancement
            metadata['has_llm_enhancement'] = 'llm_metadata' in chunk
            
            chunk_metadatas.append(metadata)
        
        # Index enriched chunks
        self.embedding_function.set_query_mode(False)
        
        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadatas
        )
        
        index_time = time.time() - start_time
        print(f"✅ Indexed {len(chunk_ids)} enriched chunks in {index_time:.2f}s")
        return collection
    
    def search(self, query: str, n_results: int = 5) -> Tuple[List[EnhancedSearchResult], float]:
        """Search using enriched embeddings."""
        start_time = time.time()
        
        # Set embedding function to query mode
        self.embedding_function.set_query_mode(True)
        
        # Search enriched content
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Reset to document mode
        self.embedding_function.set_query_mode(False)
        
        search_time = time.time() - start_time
        
        # Convert to results
        search_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            chunk_id = results['ids'][0][i]
            
            # Get original chunk for LLM metadata
            original_chunk = next((c for c in self.chunks if c['id'] == chunk_id), None)
            llm_keywords = []
            if original_chunk and 'llm_metadata' in original_chunk:
                llm_keywords = original_chunk['llm_metadata'].get('search_keywords', [])
            
            search_results.append(EnhancedSearchResult(
                chunk_id=chunk_id,
                score=results['distances'][0][i],
                chunk_type=metadata.get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",
                method="enhanced",
                llm_keywords=llm_keywords[:3] if llm_keywords else None,
                matched_on="enriched"
            ))
        
        return search_results, search_time
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Evaluate search performance."""
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
        
        print(f"\n🧪 Evaluating Enhanced Search with LLM Metadata")
        print("=" * 70)
        
        results = []
        total_relevance = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            # Search
            search_results, search_time = self.search(query)
            
            print(f"   Results ({search_time*1000:.1f}ms):")
            
            relevance_count = 0
            for i, result in enumerate(search_results[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                if result.chunk_id in expected:
                    relevance_count += 1
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
                if result.llm_keywords:
                    print(f"         LLM keywords: {', '.join(result.llm_keywords)}")
            
            # Calculate relevance
            query_relevance = relevance_count / min(len(expected), 3)
            total_relevance.append(query_relevance)
            
            results.append({
                "query": query,
                "relevance": query_relevance,
                "time": search_time
            })
        
        # Calculate averages
        avg_relevance = sum(total_relevance) / len(total_relevance)
        avg_time = sum(r["time"] for r in results) / len(results)
        
        print(f"\n📈 Final Enhanced Search Performance")
        print("=" * 50)
        print(f"Average Relevance: {avg_relevance:.1%}")
        print(f"Average Time: {avg_time*1000:.1f}ms")
        
        # Check enhanced vs non-enhanced
        enhanced_count = sum(1 for c in self.chunks if 'llm_metadata' in c)
        print(f"Chunks with LLM metadata: {enhanced_count}/{len(self.chunks)}")
        
        if avg_relevance >= 0.7:
            print("🎯 Achieved 70%+ relevance target!")
        else:
            print(f"📊 Current: {avg_relevance:.1%}, Target: 70%")
        
        return {
            "average_relevance": avg_relevance,
            "average_time": avg_time,
            "detailed_results": results
        }


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    enhanced_file = "/Users/bartosz/dev/knowledge-server/prototypes/enhanced_chunks_sample.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print("🚀 Final Enhanced Search Implementation")
    print("=" * 60)
    
    # Test with enhanced chunks
    searcher = FinalEnhancedSearch(chunks_file, enhanced_file)
    results = searcher.run_evaluation()
    
    print(f"\n🏆 Final Results Summary")
    print("=" * 40)
    print(f"Relevance: {results['average_relevance']:.1%}")
    print(f"Speed: {results['average_time']*1000:.1f}ms")
    
    # Recommendations
    print("\n💡 Next Steps:")
    if results['average_relevance'] >= 0.7:
        print("✅ 70% target achieved! Ready for production.")
        print("- Process all chunks with LLM enhancement")
        print("- Implement in MCP server")
    else:
        print("📈 Promising results! To reach 70%:")
        print("- Enhance more chunks with LLM")
        print("- Fine-tune prompts for specific chunk types")
        print("- Consider query expansion at search time")


if __name__ == "__main__":
    main()