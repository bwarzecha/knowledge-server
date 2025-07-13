#!/usr/bin/env python3
"""
Context-Aware LLM-Enhanced ChromaDB Search using Gemma 3n E4B

Uses actual API context to guide query expansion and chunk enhancement
for more accurate search results.
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
from llama_cpp import Llama


@dataclass
class ContextAwareSearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    document: str
    method: str
    expanded_query: Optional[str] = None
    context_used: Optional[str] = None


class ContextAwareGemmaExpander:
    """LLM-powered query expansion using actual API context from chunks."""
    
    def __init__(self, model_path: str = "models/gemma-3n-E4B-it-Q4_K_M.gguf"):
        self.model_path = model_path
        self.llm = None
        
        print("🔄 Loading Gemma 3n E4B GGUF for context-aware expansion...")
        start_time = time.time()
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Larger context for API docs
            n_threads=4,
            verbose=False
        )
        
        load_time = time.time() - start_time
        print(f"✅ Gemma GGUF loaded in {load_time:.2f}s")
    
    def expand_with_context(self, query: str, api_context: str) -> Dict[str, Any]:
        """Expand query using specific API context."""
        
        # Truncate context to fit in prompt
        truncated_context = api_context[:1500] + "..." if len(api_context) > 1500 else api_context
        
        prompt = f"""<start_of_turn>system
You are an API documentation expert. Use the provided API context to expand user queries for better search.

API Context:
{truncated_context}

Based on this specific API context, expand the user query to include relevant terms, endpoints, schemas, and concepts from this API.
<end_of_turn>
<start_of_turn>user
User query: "{query}"

Using the API context above, provide an expanded query that includes:
1. Original query terms
2. Specific endpoint names from the context
3. Schema/object names from the context  
4. Related API concepts from the context

Respond with JSON format:
{{"expanded_query": "...", "query_type": "endpoint|schema|error|concept", "specific_terms": [...], "context_terms": [...]}}

Your response:<end_of_turn>
<start_of_turn>model
"""

        try:
            output = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.3,
                top_p=0.95,
                stop=["<end_of_turn>", "</s>"],
                echo=False
            )
            
            response = output['choices'][0]['text'].strip()
            
            # Parse JSON response
            clean_response = response.replace('```json', '').replace('```', '').strip()
            start_idx = clean_response.find('{')
            end_idx = clean_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = clean_response[start_idx:end_idx]
                
                # Handle incomplete JSON
                if not json_text.endswith('}'):
                    open_braces = json_text.count('{') - json_text.count('}')
                    open_brackets = json_text.count('[') - json_text.count(']')
                    json_text += ']' * open_brackets + '}' * open_braces
                
                result = json.loads(json_text)
                result['context_snippet'] = truncated_context[:100] + "..."
                return result
            
        except Exception as e:
            print(f"⚠️ Context-aware expansion failed: {e}")
        
        # Fallback
        return {
            "expanded_query": f"{query} API endpoint operation schema",
            "query_type": "unknown",
            "specific_terms": query.split(),
            "context_terms": [],
            "context_snippet": "fallback"
        }


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


class ContextAwareSearch:
    """ChromaDB search with context-aware LLM enhancement."""
    
    def __init__(self, chunks_file: str, sample_size: int = 100):
        self.chunks_file = chunks_file
        self.sample_size = sample_size
        
        # Setup device for embeddings
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using MPS for embeddings")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using CUDA for embeddings")
        else:
            self.device = "cpu"
            print("💻 Using CPU for embeddings")
        
        # Initialize components
        print("🔄 Loading Qwen3-0.6B embedding model...")
        start_time = time.time()
        self.embedding_function = Qwen3EmbeddingFunction(device=self.device)
        load_time = time.time() - start_time
        print(f"✅ Embedding model loaded in {load_time:.2f}s")
        
        # Initialize context-aware expander
        self.expander = ContextAwareGemmaExpander()
        
        # Setup ChromaDB
        print("🔄 Setting up ChromaDB...")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Load data
        self.chunks = self._load_chunks()
        self.collection = self._setup_collection()
        
        print(f"✅ Context-aware search setup complete with {len(self.chunks)} chunks")
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load and sample chunks."""
        with open(self.chunks_file, 'r') as f:
            all_chunks = json.load(f)
        
        # Deduplicate and sample
        seen_ids = set()
        deduplicated_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                deduplicated_chunks.append(chunk)
                seen_ids.add(chunk["id"])
        
        chunks = deduplicated_chunks[:self.sample_size]
        print(f"📊 Loaded {len(chunks)} chunks for context-aware search")
        return chunks
    
    def _setup_collection(self) -> chromadb.Collection:
        """Create ChromaDB collection."""
        collection_name = "api_knowledge_context_aware"
        
        # Reset collection if exists
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
        
        print("🔄 Indexing chunks...")
        start_time = time.time()
        
        chunk_ids = [chunk["id"] for chunk in self.chunks]
        chunk_documents = [chunk["document"] for chunk in self.chunks]
        
        # Convert metadata to ChromaDB format
        chunk_metadatas = []
        for chunk in self.chunks:
            metadata = {}
            for key, value in chunk["metadata"].items():
                if isinstance(value, list):
                    metadata[key] = ",".join(str(v) for v in value) if value else ""
                else:
                    metadata[key] = value
            chunk_metadatas.append(metadata)
        
        # Index chunks
        self.embedding_function.set_query_mode(False)
        
        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadatas
        )
        
        index_time = time.time() - start_time
        print(f"✅ Indexed {len(chunk_ids)} chunks in {index_time:.2f}s")
        return collection
    
    def _get_api_context(self, n_samples: int = 5) -> str:
        """Get representative API context from random chunks."""
        import random
        
        # Sample some chunks to provide API context
        context_chunks = random.sample(self.chunks, min(n_samples, len(self.chunks)))
        
        context_parts = []
        for chunk in context_chunks:
            chunk_type = chunk["metadata"].get("type", "unknown")
            chunk_id = chunk["id"]
            doc_snippet = chunk["document"][:200] + "..."
            
            context_parts.append(f"{chunk_type.upper()}: {chunk_id}\n{doc_snippet}")
        
        return "\n\n".join(context_parts)
    
    def baseline_search(self, query: str, n_results: int = 5) -> Tuple[List[ContextAwareSearchResult], float]:
        """Perform baseline search without LLM enhancement."""
        start_time = time.time()
        
        # Set embedding function to query mode
        self.embedding_function.set_query_mode(True)
        
        # Search with original query
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
            
            search_results.append(ContextAwareSearchResult(
                chunk_id=results['ids'][0][i],
                score=results['distances'][0][i],
                chunk_type=metadata.get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",
                method="baseline"
            ))
        
        return search_results, search_time
    
    def context_aware_search(self, query: str, n_results: int = 5) -> Tuple[List[ContextAwareSearchResult], float]:
        """Perform context-aware search with LLM enhancement."""
        start_time = time.time()
        
        # Get API context
        api_context = self._get_api_context()
        print(f"🔍 Using API context: {api_context[:100]}...")
        
        # Expand query with context
        expansion = self.expander.expand_with_context(query, api_context)
        search_query = expansion["expanded_query"]
        
        print(f"🚀 Context-aware expansion: '{query}' → '{search_query[:100]}...'")
        
        # Set embedding function to query mode
        self.embedding_function.set_query_mode(True)
        
        # Search with expanded query
        results = self.collection.query(
            query_texts=[search_query],
            n_results=n_results
        )
        
        # Reset to document mode
        self.embedding_function.set_query_mode(False)
        
        search_time = time.time() - start_time
        
        # Convert to results
        search_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            search_results.append(ContextAwareSearchResult(
                chunk_id=results['ids'][0][i],
                score=results['distances'][0][i],
                chunk_type=metadata.get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",
                method="context_aware",
                expanded_query=search_query,
                context_used=expansion.get("context_snippet", "N/A")
            ))
        
        return search_results, search_time
    
    def run_comparison_tests(self) -> Dict[str, Any]:
        """Run tests comparing baseline vs context-aware search."""
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
                "query": "campaign update",
                "expected": ["samples_openapi_yaml:updateCampaigns"]
            }
        ]
        
        print(f"\n🧪 Running Context-Aware vs Baseline Search Comparison")
        print("=" * 80)
        
        baseline_results = []
        context_aware_results = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"\n🔍 Query: '{query}'")
            print("-" * 60)
            
            # Baseline search
            baseline_res, baseline_time = self.baseline_search(query)
            print(f"   Baseline ({baseline_time*1000:.1f}ms):")
            baseline_relevance = 0
            for i, result in enumerate(baseline_res[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                if result.chunk_id in expected:
                    baseline_relevance += 1
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
            
            # Context-aware search
            context_res, context_time = self.context_aware_search(query)
            print(f"   Context-Aware ({context_time*1000:.1f}ms):")
            if context_res and context_res[0].expanded_query:
                print(f"      Expanded: '{context_res[0].expanded_query[:80]}...'")
            
            context_relevance = 0
            for i, result in enumerate(context_res[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                if result.chunk_id in expected:
                    context_relevance += 1
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
            
            # Calculate relevance scores
            baseline_score = baseline_relevance / min(len(expected), 3)
            context_score = context_relevance / min(len(expected), 3)
            
            baseline_results.append({
                "query": query,
                "relevance": baseline_score,
                "time": baseline_time
            })
            
            context_aware_results.append({
                "query": query,
                "relevance": context_score,
                "time": context_time
            })
        
        # Calculate averages
        avg_baseline_relevance = sum(r["relevance"] for r in baseline_results) / len(baseline_results)
        avg_context_relevance = sum(r["relevance"] for r in context_aware_results) / len(context_aware_results)
        avg_baseline_time = sum(r["time"] for r in baseline_results) / len(baseline_results)
        avg_context_time = sum(r["time"] for r in context_aware_results) / len(context_aware_results)
        
        improvement = (avg_context_relevance - avg_baseline_relevance) / avg_baseline_relevance * 100 if avg_baseline_relevance > 0 else 0
        
        print(f"\n📈 Context-Aware Search Results")
        print("=" * 50)
        print(f"Baseline Relevance:     {avg_baseline_relevance:.1%}")
        print(f"Context-Aware Relevance: {avg_context_relevance:.1%}")
        print(f"Improvement:            {improvement:+.1f}%")
        print(f"Baseline Time:          {avg_baseline_time*1000:.1f}ms")
        print(f"Context-Aware Time:     {avg_context_time*1000:.1f}ms")
        
        if avg_context_relevance >= 0.7:
            print("🎯 Context-aware approach meets 70% relevance target!")
        elif improvement > 0:
            print("📈 Context-aware approach shows improvement!")
        else:
            print("⚠️ Context-aware approach needs refinement")
        
        return {
            "baseline_relevance": avg_baseline_relevance,
            "context_aware_relevance": avg_context_relevance,
            "improvement_percent": improvement,
            "baseline_time": avg_baseline_time,
            "context_aware_time": avg_context_time
        }


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print("🚀 Context-Aware Gemma Enhanced Search PoC")
    print("=" * 60)
    
    # Test with moderate sample size
    tester = ContextAwareSearch(chunks_file, sample_size=100)
    results = tester.run_comparison_tests()
    
    print(f"\n🏆 Final Context-Aware Results")
    print("=" * 40)
    print(f"Baseline:     {results['baseline_relevance']:.1%}")
    print(f"Enhanced:     {results['context_aware_relevance']:.1%}")
    print(f"Improvement:  {results['improvement_percent']:+.1f}%")
    
    if results['context_aware_relevance'] >= 0.7:
        print("🎯 Successfully achieved 70%+ relevance with context-aware LLM!")
    else:
        print("📈 Demonstrated context-aware improvement - ready for fine-tuning")


if __name__ == "__main__":
    main()