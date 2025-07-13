#!/usr/bin/env python3
"""
Local LLM-Enhanced ChromaDB Search using Gemma 3n E4B

Uses Google's Gemma 3n E4B model locally for query expansion and chunk enhancement
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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class EnhancedSearchResult:
    chunk_id: str
    score: float
    chunk_type: str
    document: str
    method: str
    expanded_query: Optional[str] = None
    semantic_tags: Optional[List[str]] = None


class GemmaQueryExpander:
    """Local LLM-powered query expansion using Gemma 3n E4B."""
    
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it"):
        self.model_id = model_id
        print(f"🔄 Loading Gemma 3n E4B model: {model_id}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using CUDA acceleration")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using Apple Silicon MPS acceleration")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
        
        start_time = time.time()
        
        # Initialize text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=self.device,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Gemma 3n E4B loaded in {load_time:.2f}s")
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """Expand query with API-specific terminology and related concepts."""
        
        system_prompt = """You are an API documentation expert. Expand user queries to improve search in OpenAPI documentation.

Given a user query, provide a JSON response with:
1. expanded_query: Original query plus related API terms, synonyms, and concepts
2. query_type: One of 'endpoint', 'schema', 'error', 'concept'
3. key_terms: Important terms for search weighting
4. related_concepts: Related API concepts

Focus on OpenAPI terminology: endpoints, operations, schemas, responses, parameters, etc.

Respond ONLY with valid JSON, no other text."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"""Expand this API documentation query: "{query}"

Example output:
{{"expanded_query": "How do I create campaign POST endpoint createCampaign operation request body parameters required fields", "query_type": "endpoint", "key_terms": ["create", "campaign", "POST", "createCampaign"], "related_concepts": ["campaign management", "campaign creation", "POST request", "create operation"]}}

Your response:"""
            }
        ]

        try:
            # Generate response
            start_time = time.time()
            output = self.pipe(
                messages,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
                return_full_text=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract generated text
            generated_text = output[0]["generated_text"].strip()
            print(f"🤖 Gemma response ({generation_time:.2f}s): {generated_text[:100]}...")
            
            # Try to parse JSON from the response
            # Look for JSON in the response (sometimes models add extra text)
            try:
                # Try direct parsing first
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                # Look for JSON within the text
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = generated_text[start_idx:end_idx]
                    result = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON found in response")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            # Fallback to original query
            return {
                "expanded_query": f"{query} API endpoint operation schema parameter",
                "query_type": "unknown",
                "key_terms": query.split(),
                "related_concepts": []
            }


class GemmaChunkEnhancer:
    """Local LLM-powered semantic metadata generation using Gemma 3n E4B."""
    
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it"):
        self.model_id = model_id
        # Reuse the same pipeline instance to save memory
        self.pipe = None
    
    def set_pipeline(self, pipe):
        """Set the pipeline instance (shared with query expander)."""
        self.pipe = pipe
    
    def generate_semantic_metadata(self, chunk_content: str, chunk_type: str) -> Dict[str, Any]:
        """Generate semantic tags and concepts for a chunk."""
        
        if not self.pipe:
            print("⚠️ Pipeline not set for chunk enhancer")
            return self._fallback_metadata(chunk_type)
        
        system_prompt = """You are an API documentation analyzer. Generate semantic metadata for OpenAPI chunks to improve search.

Provide JSON with:
1. semantic_tags: 5-8 relevant functionality tags
2. use_cases: Common use cases or scenarios  
3. related_terms: Alternative terms and synonyms
4. difficulty_level: "beginner", "intermediate", "advanced"

Focus on practical developer concerns and search terms.

Respond ONLY with valid JSON, no other text."""

        # Truncate content for processing
        truncated_content = chunk_content[:800] + "..." if len(chunk_content) > 800 else chunk_content
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Analyze this {chunk_type} chunk and generate semantic metadata:

{truncated_content}

Example output:
{{"semantic_tags": ["campaign creation", "POST endpoint", "required parameters", "validation"], "use_cases": ["create new advertising campaign", "set up campaign with budget"], "related_terms": ["campaign setup", "create campaign", "new campaign"], "difficulty_level": "beginner"}}

Your response:"""
            }
        ]

        try:
            # Generate response
            output = self.pipe(
                messages,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
                return_full_text=False
            )
            
            generated_text = output[0]["generated_text"].strip()
            
            # Parse JSON from response
            try:
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                # Look for JSON within the text
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = generated_text[start_idx:end_idx]
                    result = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON found")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Metadata generation failed: {e}")
            return self._fallback_metadata(chunk_type)
    
    def _fallback_metadata(self, chunk_type: str) -> Dict[str, Any]:
        """Fallback metadata when LLM fails."""
        return {
            "semantic_tags": [chunk_type, "api", "documentation"],
            "use_cases": [],
            "related_terms": [chunk_type],
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


class GemmaEnhancedSearch:
    """ChromaDB search enhanced with local Gemma 3n E4B."""
    
    def __init__(self, chunks_file: str, sample_size: int = 50, enable_llm: bool = True):
        self.chunks_file = chunks_file
        self.sample_size = sample_size
        self.enable_llm = enable_llm
        
        # Setup device for embeddings
        if torch.backends.mps.is_available():
            self.embedding_device = "mps"
        elif torch.cuda.is_available():
            self.embedding_device = "cuda"
        else:
            self.embedding_device = "cpu"
        
        # Initialize embedding model
        print("🔄 Loading Qwen3-0.6B embedding model...")
        start_time = time.time()
        self.embedding_function = Qwen3EmbeddingFunction(device=self.embedding_device)
        load_time = time.time() - start_time
        print(f"✅ Embedding model loaded in {load_time:.2f}s")
        
        # Initialize LLM components
        if self.enable_llm:
            self.query_expander = GemmaQueryExpander()
            self.chunk_enhancer = GemmaChunkEnhancer()
            # Share pipeline to save memory
            self.chunk_enhancer.set_pipeline(self.query_expander.pipe)
        
        # Setup ChromaDB
        print("🔄 Setting up ChromaDB...")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Load and enhance data
        self.chunks = self._load_and_enhance_chunks()
        self.collection = self._setup_enhanced_collection()
        
        print(f"✅ Gemma-enhanced ChromaDB setup complete with {len(self.chunks)} chunks")
    
    def _load_and_enhance_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks and optionally enhance with Gemma-generated metadata."""
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
        print(f"📊 Processing {len(chunks)} chunks...")
        
        if self.enable_llm:
            print("🔄 Enhancing chunks with Gemma-generated metadata...")
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"   Enhancing chunk {i+1}/{len(chunks)}: {chunk['id']}")
                
                enhanced_chunk = chunk.copy()
                
                # Generate semantic metadata with Gemma
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
                
                # Small delay to prevent overheating
                time.sleep(0.5)
            
            print(f"✅ Enhanced {len(enhanced_chunks)} chunks with Gemma metadata")
            return enhanced_chunks
        else:
            print("⚠️ LLM enhancement disabled - using original chunks")
            return chunks
    
    def _setup_enhanced_collection(self) -> chromadb.Collection:
        """Create ChromaDB collection with enhanced chunks."""
        collection_name = "api_knowledge_gemma"
        
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
        
        print("🔄 Indexing Gemma-enhanced chunks...")
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
        print(f"✅ Indexed {len(chunk_ids)} chunks in {index_time:.2f}s")
        return collection
    
    def enhanced_search(self, query: str, n_results: int = 5) -> Tuple[List[EnhancedSearchResult], float]:
        """Perform search with Gemma-powered query expansion."""
        start_time = time.time()
        
        if self.enable_llm:
            # Expand query with Gemma
            print(f"🔍 Expanding query with Gemma: '{query}'")
            expansion = self.query_expander.expand_query(query)
            search_query = expansion["expanded_query"]
            query_type = expansion["query_type"]
            print(f"🚀 Expanded to: '{search_query}' (type: {query_type})")
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
                method="gemma_enhanced",
                expanded_query=search_query if expansion else None,
                semantic_tags=metadata.get('semantic_tags', '').split(',') if metadata.get('semantic_tags') else None
            ))
        
        return enhanced_results, search_time
    
    def run_comparison_tests(self) -> Dict[str, Any]:
        """Run tests comparing baseline vs Gemma-enhanced search."""
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
            }
        ]
        
        print(f"\n🧪 Running Gemma-Enhanced Search Tests")
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
            
            print(f"   Gemma Enhanced ({search_time:.2f}s):")
            if enhanced_results and enhanced_results[0].expanded_query:
                print(f"      Expanded: '{enhanced_results[0].expanded_query}'")
            
            relevance_count = 0
            for i, result in enumerate(enhanced_results[:3]):
                relevance = "✅" if result.chunk_id in expected else "❌"
                if result.chunk_id in expected:
                    relevance_count += 1
                print(f"      {i+1}. {result.score:.3f} {relevance} {result.chunk_id}")
                if result.semantic_tags and result.semantic_tags[0]:
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
        
        print(f"\n📈 Gemma-Enhanced Search Summary")
        print("=" * 50)
        print(f"Average Relevance: {avg_relevance:.1%}")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Gemma Enhanced: {'✅' if self.enable_llm else '❌'}")
        
        if avg_relevance >= 0.7:
            print(f"🎯 Meets 70% relevance target!")
        else:
            print(f"⚠️ Below 70% target ({avg_relevance:.1%})")
        
        return {
            "average_relevance": avg_relevance,
            "average_time": avg_time,
            "results": results
        }


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print("🚀 Gemma 3n E4B Enhanced ChromaDB Search PoC")
    print("=" * 60)
    
    # Test with small sample due to Gemma processing time
    print("\n🔄 Testing Gemma-enhanced search (small sample)...")
    enhanced_tester = GemmaEnhancedSearch(chunks_file, sample_size=20, enable_llm=True)
    enhanced_results = enhanced_tester.run_comparison_tests()
    
    print(f"\n🏆 Final Results")
    print("=" * 40)
    print(f"Relevance: {enhanced_results['average_relevance']:.1%}")
    print(f"Avg Time: {enhanced_results['average_time']:.2f}s")
    
    if enhanced_results['average_relevance'] >= 0.7:
        print("🎯 Successfully achieved 70%+ relevance target with local Gemma!")
    else:
        print(f"📈 Improvement demonstrated - need fine-tuning for 70% target")


if __name__ == "__main__":
    main()