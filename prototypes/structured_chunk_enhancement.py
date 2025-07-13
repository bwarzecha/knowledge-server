#!/usr/bin/env python3
"""
Structured Chunk Enhancement with Clear Context Instructions

Enhances chunks with LLM by providing clear instructions about:
- What chunk is being enhanced
- Its position in the API hierarchy
- What related information is available
"""

import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from llama_cpp import Llama


@dataclass
class ChunkContext:
    """Structured context for a chunk."""
    chunk_id: str
    chunk_type: str  # endpoint, schema, error
    chunk_content: str
    parent_endpoint: Optional[str] = None  # For schemas used in endpoints
    related_schemas: Optional[List[str]] = None  # Schemas referenced
    http_method: Optional[str] = None  # GET, POST, etc.
    path: Optional[str] = None  # API path
    operation_id: Optional[str] = None


class StructuredChunkEnhancer:
    """Enhance chunks with structured context understanding."""
    
    def __init__(self, model_path: str = "models/gemma-3n-E4B-it-Q4_K_M.gguf"):
        self.model_path = model_path
        print("🔄 Loading Gemma for structured chunk enhancement...")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
        print("✅ Gemma loaded for structured enhancement")
    
    def enhance_chunk(self, context: ChunkContext) -> Dict[str, Any]:
        """Enhance a single chunk with structured metadata."""
        
        # Build structured prompt based on chunk type
        if context.chunk_type == "endpoint":
            prompt = self._build_endpoint_prompt(context)
        elif context.chunk_type == "schema":
            prompt = self._build_schema_prompt(context)
        else:
            prompt = self._build_generic_prompt(context)
        
        try:
            output = self.llm(
                prompt,
                max_tokens=400,
                temperature=0.2,  # Lower temperature for consistency
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
                
                # Fix incomplete JSON
                if not json_text.endswith('}'):
                    open_braces = json_text.count('{') - json_text.count('}')
                    open_brackets = json_text.count('[') - json_text.count(']')
                    json_text += ']' * open_brackets + '}' * open_braces
                
                result = json.loads(json_text)
                result['chunk_id'] = context.chunk_id
                result['enhancement_success'] = True
                return result
                
        except Exception as e:
            print(f"⚠️ Enhancement failed for {context.chunk_id}: {e}")
        
        # Fallback
        return {
            'chunk_id': context.chunk_id,
            'enhancement_success': False,
            'search_keywords': [],
            'semantic_context': [],
            'usage_patterns': []
        }
    
    def _build_endpoint_prompt(self, context: ChunkContext) -> str:
        """Build prompt for endpoint chunks."""
        
        related_info = ""
        if context.related_schemas:
            related_info = f"\nThis endpoint uses these schemas: {', '.join(context.related_schemas)}"
        
        prompt = f"""<start_of_turn>system
You are enhancing API documentation chunks for better search. You must understand the chunk's role and relationships.

CURRENT CHUNK BEING ENHANCED:
- Type: API Endpoint
- ID: {context.chunk_id}
- HTTP Method: {context.http_method or 'Unknown'}
- Path: {context.path or 'Unknown'}
- Operation: {context.operation_id or 'Unknown'}
{related_info}

INSTRUCTIONS:
1. This is an ENDPOINT chunk - focus on what this endpoint DOES
2. Include the actual endpoint path and method in search terms
3. Consider what developers would search for to find THIS specific endpoint
4. Include both technical terms (POST, GET) and business terms (create, update, list)
<end_of_turn>
<start_of_turn>user
Enhance this endpoint chunk for search:

{context.chunk_content[:800]}

Generate metadata in this JSON format:
{{
  "search_keywords": ["exact terms developers would search"],
  "semantic_context": ["what this endpoint accomplishes"],
  "usage_patterns": ["when developers would use this"],
  "alternative_queries": ["other ways to ask for this endpoint"]
}}

Your response:<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _build_schema_prompt(self, context: ChunkContext) -> str:
        """Build prompt for schema chunks."""
        
        parent_info = ""
        if context.parent_endpoint:
            parent_info = f"\nThis schema is used by endpoint: {context.parent_endpoint}"
        
        prompt = f"""<start_of_turn>system
You are enhancing API documentation chunks for better search. You must understand the chunk's role and relationships.

CURRENT CHUNK BEING ENHANCED:
- Type: Data Schema/Model
- ID: {context.chunk_id}
{parent_info}

INSTRUCTIONS:
1. This is a SCHEMA chunk - focus on the data structure and fields
2. Include the actual schema name in search terms
3. Consider what properties/fields developers would search for
4. Link to parent endpoints if known
<end_of_turn>
<start_of_turn>user
Enhance this schema chunk for search:

{context.chunk_content[:800]}

Generate metadata in this JSON format:
{{
  "search_keywords": ["schema name", "key properties", "field names"],
  "semantic_context": ["what this schema represents"],
  "usage_patterns": ["when this schema is used"],
  "related_operations": ["operations that use this schema"]
}}

Your response:<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _build_generic_prompt(self, context: ChunkContext) -> str:
        """Build prompt for other chunk types."""
        
        prompt = f"""<start_of_turn>system
You are enhancing API documentation chunks for better search.

CURRENT CHUNK BEING ENHANCED:
- Type: {context.chunk_type}
- ID: {context.chunk_id}

INSTRUCTIONS:
1. Focus on what this chunk documents
2. Extract key terms and concepts
3. Consider developer search intent
<end_of_turn>
<start_of_turn>user
Enhance this chunk for search:

{context.chunk_content[:800]}

Generate metadata in this JSON format:
{{
  "search_keywords": ["key terms from content"],
  "semantic_context": ["what this describes"],
  "usage_patterns": ["when this is relevant"],
  "category": ["type of documentation"]
}}

Your response:<end_of_turn>
<start_of_turn>model
"""
        return prompt


def test_structured_enhancement():
    """Test structured chunk enhancement with sample data."""
    
    enhancer = StructuredChunkEnhancer()
    
    # Load sample chunks
    chunks_file = "/Users/bartosz/dev/knowledge-server/prototypes/chunks_export.json"
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)[:5]  # Just test with first 5
    
    print("\n🧪 Testing Structured Chunk Enhancement")
    print("=" * 70)
    
    enhanced_chunks = []
    
    for chunk in chunks:
        print(f"\n📄 Enhancing: {chunk['id']}")
        print(f"   Type: {chunk['metadata'].get('type', 'unknown')}")
        
        # Build structured context
        context = ChunkContext(
            chunk_id=chunk['id'],
            chunk_type=chunk['metadata'].get('type', 'unknown'),
            chunk_content=chunk['document'],
            parent_endpoint=chunk['metadata'].get('parent_endpoint'),
            related_schemas=chunk['metadata'].get('ref_ids', []),
            http_method=chunk['metadata'].get('method'),
            path=chunk['metadata'].get('path'),
            operation_id=chunk['metadata'].get('operationId')
        )
        
        # Enhance chunk
        start_time = time.time()
        enhancement = enhancer.enhance_chunk(context)
        enhance_time = time.time() - start_time
        
        print(f"   Time: {enhance_time:.2f}s")
        
        if enhancement.get('enhancement_success'):
            print(f"   ✅ Enhanced successfully")
            print(f"   Keywords: {enhancement.get('search_keywords', [])[:3]}")
            print(f"   Context: {enhancement.get('semantic_context', [])[:2]}")
            
            # Merge enhancement with original chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk['llm_metadata'] = enhancement
            enhanced_chunks.append(enhanced_chunk)
        else:
            print(f"   ❌ Enhancement failed")
    
    # Save enhanced chunks
    output_file = "prototypes/enhanced_chunks_sample.json"
    with open(output_file, 'w') as f:
        json.dump(enhanced_chunks, f, indent=2)
    
    print(f"\n✅ Saved {len(enhanced_chunks)} enhanced chunks to {output_file}")
    
    # Show example of how to use in search
    print("\n💡 Example: How enhanced metadata improves search")
    print("-" * 50)
    
    if enhanced_chunks:
        example = enhanced_chunks[0]
        print(f"Original chunk: {example['id']}")
        print(f"Original content: {example['document'][:100]}...")
        
        if 'llm_metadata' in example:
            meta = example['llm_metadata']
            print(f"\nEnhanced with:")
            print(f"- Search keywords: {meta.get('search_keywords', [])}")
            print(f"- Semantic context: {meta.get('semantic_context', [])}")
            print(f"- Usage patterns: {meta.get('usage_patterns', [])}")


def main():
    """Run structured enhancement test."""
    test_structured_enhancement()


if __name__ == "__main__":
    main()