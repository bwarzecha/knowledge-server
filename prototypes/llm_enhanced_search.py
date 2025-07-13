#!/usr/bin/env python3
"""
LLM-Enhanced ChromaDB Search

Adds LLM-powered query expansion and chunk metadata enhancement
to improve search relevance from 60% to 70%+.
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
import boto3


@dataclass
class EnhancedSearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    document: str
    method: str
    expanded_query: Optional[str] = None
    semantic_tags: Optional[List[str]] = None


class QueryExpander:
    """LLM-powered query expansion for better retrieval using AWS Bedrock."""
    
    def __init__(self, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime')
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """Expand query with API-specific terminology and related concepts."""
        
        system_prompt = """You are an API documentation expert. Your task is to expand user queries to improve search in OpenAPI documentation.

Given a user query, provide:
1. expanded_query: The original query plus related API terms, synonyms, and concepts
2. query_type: One of 'endpoint', 'schema', 'error', 'concept'
3. key_terms: Important terms that should be weighted higher in search
4. related_concepts: Related API concepts that might be relevant

Focus on API/OpenAPI terminology like endpoints, operations, schemas, responses, parameters, etc.

Respond with valid JSON only."""

        user_prompt = f"""Expand this API documentation query: "{query}"

Example:
Query: "How do I create a campaign?"
Response: {{
  "expanded_query": "How do I create campaign POST endpoint createCampaign operation request body parameters required fields",
  "query_type": "endpoint", 
  "key_terms": ["create", "campaign", "POST", "createCampaign"],
  "related_concepts": ["campaign management", "campaign creation", "POST request", "create operation"]
}}

Now expand: "{query}" """

        try:
            # Prepare request for Claude 3.5 Haiku
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "temperature": 0.3,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text'].strip()
            
            result = json.loads(content)
            return result
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            # Fallback to original query
            return {
                "expanded_query": query,
                "query_type": "unknown",
                "key_terms": query.split(),
                "related_concepts": []
            }


class ChunkEnhancer:
    """LLM-powered semantic metadata generation for chunks using AWS Bedrock."""
    
    def __init__(self, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime')
    
    def generate_semantic_metadata(self, chunk_content: str, chunk_type: str) -> Dict[str, Any]:
        """Generate semantic tags and concepts for a chunk."""
        
        system_prompt = """You are an API documentation analyzer. Generate semantic metadata for OpenAPI documentation chunks to improve search relevance.

For each chunk, provide:
1. semantic_tags: 5-8 relevant tags describing the functionality
2. use_cases: Common use cases or scenarios  
3. related_terms: Alternative terms and synonyms
4. difficulty_level: "beginner", "intermediate", "advanced"

Focus on practical developer concerns and search terms they might use.

Respond with valid JSON only."""

        # Truncate content for LLM processing
        truncated_content = chunk_content[:1000] + "..." if len(chunk_content) > 1000 else chunk_content
        
        user_prompt = f"""Analyze this {chunk_type} chunk and generate semantic metadata:

{truncated_content}

Example response:
{{
  "semantic_tags": ["campaign creation", "POST endpoint", "required parameters", "validation"],
  "use_cases": ["create new advertising campaign", "set up campaign with budget"],
  "related_terms": ["campaign setup", "create campaign", "new campaign", "campaign management"],
  "difficulty_level": "beginner"
}}

Generate metadata for the above chunk:"""

        try:
            # Prepare request for Claude 3.5 Haiku
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.3,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text'].strip()
            
            result = json.loads(content)
            return result
            
        except Exception as e:
            print(f"⚠️ Metadata generation failed: {e}")
            # Fallback to basic metadata
            return {
                "semantic_tags": [chunk_type],
                "use_cases": [],
                "related_terms": [],
                "difficulty_level": "unknown"
            }


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for Qwen3-0.6B model with proper query prompts."""
    
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


class LLMEnhancedSearch:
    """ChromaDB search enhanced with LLM-powered query expansion and metadata."""
    
    def __init__(self, chunks_file: str, sample_size: int = 200, enable_llm: bool = True):
        self.chunks_file = chunks_file
        self.sample_size = sample_size
        self.enable_llm = enable_llm
        
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
        
        # Initialize components
        print("🔄 Loading Qwen3-0.6B embedding model...")
        start_time = time.time()
        self.embedding_function = Qwen3EmbeddingFunction(device=self.device)
        load_time = time.time() - start_time
        print(f"✅ Embedding model loaded in {load_time:.2f}s")
        
        if self.enable_llm:
            print("🔄 Initializing LLM components...")
            self.query_expander = QueryExpander()
            self.chunk_enhancer = ChunkEnhancer()
            print("✅ LLM components ready")
        
        # Setup ChromaDB
        print("🔄 Setting up ChromaDB...")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Load and enhance data
        self.chunks = self._load_and_enhance_chunks()
        self.collection = self._setup_enhanced_collection()
        
        print(f"✅ Enhanced ChromaDB setup complete with {len(self.chunks)} chunks")
    
    def _load_and_enhance_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks and optionally enhance with LLM-generated metadata."""
        with open(self.chunks_file, 'r') as f:
            all_chunks = json.load(f)
        
        # Deduplicate
        seen_ids = set()
        deduplicated_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                deduplicated_chunks.append(chunk)
                seen_ids.add(chunk["id"])
        
        # Sample for testing
        chunks = deduplicated_chunks[:self.sample_size]
        print(f"📊 Processing {len(chunks)} chunks...")
        
        if self.enable_llm:
            print("🔄 Enhancing chunks with LLM-generated metadata...")
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i % 50 == 0:
                    print(f"   Enhanced {i}/{len(chunks)} chunks...")
                
                enhanced_chunk = chunk.copy()
                
                # Generate semantic metadata
                semantic_meta = self.chunk_enhancer.generate_semantic_metadata(
                    chunk["document"], 
                    chunk["metadata"].get("type", "unknown")
                )
                
                # Add to metadata
                enhanced_chunk["metadata"]["semantic_tags"] = semantic_meta.get("semantic_tags", [])
                enhanced_chunk["metadata"]["use_cases"] = semantic_meta.get("use_cases", [])
                enhanced_chunk["metadata"]["related_terms"] = semantic_meta.get("related_terms", [])
                enhanced_chunk["metadata"]["difficulty_level"] = semantic_meta.get("difficulty_level", "unknown")
                
                enhanced_chunks.append(enhanced_chunk)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
            
            print(f"✅ Enhanced {len(enhanced_chunks)} chunks with semantic metadata")
            return enhanced_chunks
        else:
            print("⚠️ LLM enhancement disabled - using original chunks")
            return chunks
    
    def _setup_enhanced_collection(self) -> chromadb.Collection:
        """Create ChromaDB collection with enhanced chunks."""
        collection_name = "api_knowledge_enhanced"
        
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
        
        print("🔄 Indexing enhanced chunks...")
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
        
        # Index with document mode
        self.embedding_function.set_query_mode(False)
        
        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadatas
        )
        
        index_time = time.time() - start_time
        print(f"✅ Indexed {len(chunk_ids)} enhanced chunks in {index_time:.2f}s")
        return collection
    
    def enhanced_search(self, query: str, n_results: int = 5) -> Tuple[List[EnhancedSearchResult], float]:
        """Perform search with LLM-powered query expansion."""
        start_time = time.time()
        
        if self.enable_llm:
            # Expand query with LLM
            expansion = self.query_expander.expand_query(query)
            search_query = expansion["expanded_query"]
            query_type = expansion["query_type"]
            print(f"🔍 Expanded: '{query}' → '{search_query}' ({query_type})")
        else:
            search_query = query
            expansion = None
        
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
        
        # Convert to enhanced results
        enhanced_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            enhanced_results.append(EnhancedSearchResult(
                chunk_id=results['ids'][0][i],
                score=results['distances'][0][i],
                chunk_type=metadata.get('type', 'unknown'),
                document=results['documents'][0][i][:100] + "...",
                method="llm_enhanced",
                expanded_query=search_query if expansion else None,
                semantic_tags=metadata.get('semantic_tags', '').split(',') if metadata.get('semantic_tags') else None
            ))
        
        return enhanced_results, search_time
    
    def run_enhanced_tests(self) -> List[Dict[str, Any]]:
        """Run tests comparing original vs LLM-enhanced search."""
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
        
        print(f"\n🧪 Running Enhanced Search Tests")
        print("=" * 70)
        
        results = []
        total_relevance = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            # Enhanced search
            enhanced_results, search_time = self.enhanced_search(query)
            
            print(f"   Enhanced ({search_time*1000:.1f}ms):")
            if enhanced_results and enhanced_results[0].expanded_query:
                print(f"      Expanded: '{enhanced_results[0].expanded_query}'")
            
            relevance_count = 0
            for i, result in enumerate(enhanced_results[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                if result.chunk_id in expected:
                    relevance_count += 1
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
                if result.semantic_tags:
                    print(f"         Tags: {', '.join(result.semantic_tags[:3])}")
            
            # Calculate relevance
            query_relevance = relevance_count / min(len(expected), 3)
            total_relevance.append(query_relevance)
            
            results.append({
                "query": query,
                "results": enhanced_results,
                "relevance": query_relevance,
                "search_time": search_time
            })
        
        # Print summary
        avg_relevance = sum(total_relevance) / len(total_relevance)
        avg_time = sum(r["search_time"] for r in results) / len(results)
        
        print(f"\n📈 Enhanced Search Summary")
        print("=" * 50)
        print(f"Average Relevance: {avg_relevance:.1%}")
        print(f"Average Time: {avg_time*1000:.1f}ms")
        print(f"LLM Enhanced: {'✅' if self.enable_llm else '❌'}")
        
        if avg_relevance >= 0.7:
            print(f"🎯 Meets 70% relevance target!")
        else:
            print(f"⚠️ Below 70% target ({avg_relevance:.1%})")
        
        return results


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print("🚀 LLM-Enhanced ChromaDB Search Test")
    print("=" * 60)
    
    # Test with and without LLM enhancement
    print("\n🔄 Testing WITHOUT LLM enhancement...")
    baseline_tester = LLMEnhancedSearch(chunks_file, sample_size=200, enable_llm=False)
    baseline_results = baseline_tester.run_enhanced_tests()
    
    print("\n🔄 Testing WITH LLM enhancement...")
    enhanced_tester = LLMEnhancedSearch(chunks_file, sample_size=50, enable_llm=True) 
    enhanced_results = enhanced_tester.run_enhanced_tests()
    
    # Compare results
    baseline_relevance = sum(r["relevance"] for r in baseline_results) / len(baseline_results)
    enhanced_relevance = sum(r["relevance"] for r in enhanced_results) / len(enhanced_results)
    
    improvement = (enhanced_relevance - baseline_relevance) / baseline_relevance * 100
    
    print(f"\n🏆 Final Comparison")
    print("=" * 40)
    print(f"Baseline:  {baseline_relevance:.1%}")
    print(f"Enhanced:  {enhanced_relevance:.1%}")
    print(f"Improvement: {improvement:+.1f}%")
    
    if enhanced_relevance >= 0.7:
        print("🎯 Target achieved!")
    else:
        print("❌ Still below 70% target")


if __name__ == "__main__":
    main()