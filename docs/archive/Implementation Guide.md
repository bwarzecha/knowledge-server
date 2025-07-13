# OpenAPI Knowledge Server - Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the OpenAPI Knowledge Server based on our validated design that achieved 80% query completeness with 2,079 real-world chunks.

## Core Components to Build

### 1. OpenAPI Chunker (✅ Prototyped)
```python
# Key functions already implemented:
- Load OpenAPI specs (YAML/JSON)
- Build dependency graph from $ref analysis  
- Generate file-path-based chunk IDs
- Create endpoint and schema chunks
- Resolve references to metadata
```

### 2. ChromaDB Integration
```python
# Next to implement:
- Create single collection with metadata filtering
- Store chunks with validated metadata structure
- Implement batch insertion for performance
- Add metadata indexes for fast filtering
```

### 3. Multi-Stage Retrieval Pipeline
```python
# Pipeline stages to implement:
1. Semantic search → top-k chunks
2. Reference expansion → get linked schemas via ref_ids
3. Context assembly → order and deduplicate
```

### 4. MCP Server Interface
```python
# Simple API to implement:
async def askAPI(query: str) -> str:
    # 1. Retrieve relevant chunks
    # 2. Assemble context
    # 3. Query LLM (Haiku) with context
    # 4. Return answer
```

## Implementation Order

### Phase 1: Storage Layer (1-2 days)
1. Set up ChromaDB with proper collections
2. Migrate prototype chunker to production code
3. Index all OpenAPI specs from samples/
4. Verify chunk counts and metadata

### Phase 2: Retrieval Pipeline (2-3 days)
1. Implement semantic search with embeddings
2. Add reference expansion logic
3. Build context assembly with deduplication
4. Test with queries from analyzer script

### Phase 3: MCP Integration (2-3 days)
1. Create MCP server scaffold
2. Implement askAPI() endpoint
3. Integrate with Claude Haiku for answer generation
4. Add error handling and logging

### Phase 4: Enhancements (1-2 days)
1. Add API-level feature tagging
2. Improve error documentation chunks
3. Implement query caching
4. Add confidence scoring

## Key Design Decisions

### ID Generation Strategy
```python
def generate_chunk_id(file_path, natural_name):
    # Convert: /samples/openapi.yaml + "listCampaigns"
    # To: "samples_openapi_yaml:listCampaigns"
    relative_path = file_path.relative_to(knowledge_root)
    prefix = str(relative_path).replace('/', '_').replace('.', '_')
    return f"{prefix}:{natural_name}"
```

### Metadata Structure
```json
{
  "type": "endpoint|schema",
  "source_file": "samples/openapi.yaml",
  "path": "/sd/campaigns",
  "method": "GET",
  "operationId": "listCampaigns",
  "ref_ids": ["samples_openapi_yaml:Campaign", "samples_openapi_yaml:Error"],
  "tags": ["Campaigns"]
}
```

### Chunking Rules
- **Endpoint chunks**: Include operation + 1-2 levels of inlined schemas
- **Schema chunks**: Separate storage for complex/reusable schemas
- **Size target**: 200-500 tokens per chunk
- **Reference handling**: Store IDs in metadata, not content

## Testing Strategy

### Use Validated Test Queries
The analyzer script contains 20 real-world queries that achieved:
- 10 queries with ≥90% completeness
- 5 queries with 70-89% completeness  
- 5 queries needing improvement

Focus on maintaining the 80% baseline while improving:
- Cross-API feature discovery
- Error code documentation
- Complex relationship queries

### Performance Targets
- **Chunk retrieval**: <100ms for top-5 chunks
- **Reference expansion**: <50ms for 15 references
- **Total response time**: <2 seconds end-to-end
- **Token usage**: Keep under 4,000 tokens per query

## Configuration

### Embedding Model Selection (✅ Validated)
Based on comprehensive testing of 3 models against 5 test queries:

```python
EMBEDDING_MODELS = {
    "primary": "Qwen/Qwen3-Embedding-0.6B",  # 70% relevance, 141ms query time
    "alternative": "Alibaba-NLP/gte-large-en-v1.5",  # 70% relevance, 197ms query time 
    "baseline": "all-MiniLM-L6-v2"  # 30% relevance, 157ms query time
}
```

**Validation Results:**
- **Qwen3-0.6B**: Best speed/accuracy balance with MPS acceleration
- **GTE-large**: Similar accuracy but 40% slower
- **8B models**: Tested but provide marginal improvement vs resource cost
```

### LLM Configuration
```python
LLM_CONFIG = {
    "model": "claude-3-haiku",
    "max_context_tokens": 16000,
    "temperature": 0.1,
    "system_prompt": "You are an API expert. Answer based only on provided context."
}
```

## Common Pitfalls to Avoid

1. **Don't over-chunk**: Our 200-500 token size is validated
2. **Don't lose references**: Always preserve ref_ids in metadata
3. **Don't duplicate unnecessarily**: Use reference expansion, not copying
4. **Don't trust operationId uniqueness**: Always use file path prefixes
5. **Don't skip validation**: Test with the analyzer script regularly

## Success Metrics

Track these metrics to ensure implementation quality:
- **Query completeness**: Maintain ≥80% average
- **Chunk count**: ~300-400 chunks per large spec
- **Reference resolution**: 100% of ref_ids should resolve
- **Token efficiency**: Average 1,000-1,500 tokens per query
- **Response accuracy**: Validated through test suite

## Critical Areas Requiring Prototyping/Clarification

Before starting full implementation, these integration points need validation:

### 1. ChromaDB Integration Testing (High Priority)
```python
# Test this exact pattern:
collection.add(
    ids=["samples_openapi_yaml:listCampaigns"],
    documents=["GET /campaigns - retrieve campaigns..."],
    metadatas=[{"type": "endpoint", "ref_ids": ["samples_openapi_yaml:Campaign"]}]
)

# Validate performance:
collection.get(ids=ref_ids)  # Multi-stage retrieval lookup
collection.query(query_texts=["campaign"], n_results=5)  # Semantic search
```

**Questions to Answer**:
- Can ChromaDB handle our nested metadata structure efficiently?
- How fast are ID-based lookups for reference expansion?
- Does metadata filtering work with our schema?
- What's the optimal batch size for indexing 2,000+ chunks?

### 2. MCP Server Integration (High Priority)
```python
# Need basic MCP server prototype:
@mcp_tool
async def askAPI(query: str) -> str:
    chunks = retriever.get_relevant_chunks(query)
    context = assembler.build_context(chunks)
    answer = llm.generate(context, query)
    return answer
```

**Questions to Answer**:
- What's the exact MCP interface pattern?
- How to handle async operations?
- Error handling and timeout patterns?
- How to structure the MCP server project?

### 3. Context Assembly Strategy (Medium Priority)
```python
# Test different assembly approaches:
def assemble_context(primary_chunks, referenced_chunks):
    # Option 1: Primary first, then references
    # Option 2: Interleave by relevance score
    # Option 3: Group by type (endpoints, then schemas)
    pass
```

**Questions to Answer**:
- How to order chunks for optimal LLM understanding?
- How to label sections ("Endpoint:", "Schema:", etc.)?
- Deduplication strategy when same schema appears multiple times?
- How much of each chunk to include (full vs summary)?

### 4. Embedding Model Validation (✅ Completed)
**Results from comprehensive testing:**

```python
# Final validation scores against manual relevance:
Qwen3-0.6B:    70% relevance, 141.4ms avg query time
GTE-large:     70% relevance, 197.1ms avg query time  
MiniLM-L6-v2:  30% relevance, 157.7ms avg query time

# Key learnings:
- Qwen3 requires prompt_name="query" for proper usage
- Apple Silicon MPS acceleration works well
- 8B models don't justify the resource cost
```

**Implementation Notes:**
- Use `model.encode([query], prompt_name="query")` for Qwen3
- Enable MPS device detection for Apple Silicon
- Qwen3-0.6B provides best speed/accuracy trade-off

### 5. LLM Integration Details (High Priority)
```python
# Test different prompting strategies:
SYSTEM_PROMPTS = {
    "basic": "You are an API expert. Answer based only on provided context.",
    "structured": "You are an API documentation assistant. Use the provided OpenAPI context to give accurate, complete answers. Include code examples when relevant.",
    "detailed": "You are an expert API consultant. Analyze the provided OpenAPI documentation chunks and provide comprehensive answers..."
}
```

**Questions to Answer**:
- Claude Haiku vs Sonnet for this use case?
- Optimal system prompt for API documentation?
- How to handle context window limits (16k for Haiku)?
- Response formatting guidelines?

## Recommended Prototyping Order

**Week 1**: 
1. **ChromaDB Integration** - Test exact metadata structure and query patterns
2. **MCP Server Scaffold** - Get basic `askAPI()` function working end-to-end
3. **Context Assembly** - Test different chunk ordering and formatting strategies

**Week 2**:
4. **Embedding Model Testing** - Validate Qwen2.5 vs alternatives with real queries
5. **LLM Integration** - Test prompting strategies and response quality
6. **End-to-End Validation** - Full pipeline test with analyzer script queries

## Next Steps

1. Start with ChromaDB integration testing using our validated metadata structure
2. Build minimal MCP server prototype for `askAPI()` interface
3. Test context assembly strategies with our existing chunks
4. Validate each integration point before moving to production implementation
5. Use the analyzer script to test each prototype iteration

The chunking strategy is proven (80% completeness). The focus now is validating the integration layers before full system build.