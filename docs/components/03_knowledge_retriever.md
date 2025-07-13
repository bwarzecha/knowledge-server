# Knowledge Retriever Component Specification

## Component Purpose

The Knowledge Retriever is the **core intelligence** of the system. When given a query like "How do I create a user?", it doesn't just find the relevant API endpoint - it automatically discovers and includes all the related schemas, error responses, and dependencies needed to give a complete answer.

Think of it as a **smart librarian** that not only finds the book you asked for, but also gathers all the reference materials, appendices, and related documents you'll need to fully understand the topic.

## The Core Problem Being Solved

**Problem**: OpenAPI specifications have complex interdependencies. An endpoint references schemas, which reference other schemas, which have examples and error responses. A simple query like "create user" might need:

1. The POST /users endpoint definition
2. The User schema it accepts  
3. The Address schema that User references
4. The error response schemas (400, 500, etc.)
5. Related pagination or authentication schemas

**Solution**: Two-stage intelligent retrieval that finds relevant content first, then automatically expands to include all dependencies.

## How It Works: The Two-Stage Pipeline

### Stage 1: Semantic Search
**Input**: Natural language query ("How to create a user?")
**Process**: 
- Uses the Vector Store Manager to find chunks semantically similar to the query
- Returns ranked results based on embedding similarity
- No filtering - cast a wide net to avoid missing relevant content

**Output**: List of primary chunks (typically 3-5 most relevant pieces)

**Example**: Query "create user" might return:
- `api.json:paths/users/post` (POST /users endpoint)
- `api.json:components/schemas/User` (User schema definition)  
- `api.json:paths/users/get` (GET /users endpoint - also relevant)

### Stage 2: Reference Expansion
**Input**: Primary chunks from Stage 1
**Process**:
- Examines the `ref_ids` field in each chunk's metadata
- Uses breadth-first traversal to follow dependency chains
- Retrieves referenced chunks using Vector Store Manager's `get_by_ids()`
- Prevents infinite loops with visited tracking
- Limits depth to avoid context explosion

**Output**: List of referenced chunks (schemas, error responses, examples)

**Key Data Flow**:
1. Primary chunk `api.json:paths/users/post` has metadata:
   ```json
   {
     "ref_ids": {
       "api.json:components/schemas/User": [],
       "api.json:components/schemas/CreateUserRequest": [],
       "api.json:paths/users/post:errors": []
     }
   }
   ```

2. Retriever fetches these 3 referenced chunks
3. Each referenced chunk may have its own `ref_ids` (e.g., User schema references Address schema)
4. Process continues up to max_depth (default: 3 levels)

### Stage 3: Context Assembly
**Input**: Primary chunks + Referenced chunks
**Process**:
- Combines all chunks into a single `KnowledgeContext` object
- Estimates total token count using tiktoken (same as embedding utils)
- Calculates performance metrics and statistics
- Orders chunks by relevance (primary results first, then dependencies)

**Output**: Complete `KnowledgeContext` ready for LLM processing

## Key Data Structures Used

### chunk["metadata"]["ref_ids"]
**Format**: `{"referenced_chunk_id": [additional_context], ...}`
**Created by**: OpenAPI Processor during chunk creation and splitting
**Used by**: Reference expander to follow dependency chains

**Example**:
```json
{
  "ref_ids": {
    "api.json:components/schemas/Address": [],
    "api.json:components/schemas/User:examples": []
  }
}
```

### chunk["metadata"]["referenced_by"] 
**Format**: `["chunk_id_that_references_this", ...]`
**Created by**: Graph builder and chunk assembler for bidirectional tracking
**Used by**: Potential future optimizations (not core to current retrieval)

### KnowledgeContext Output Structure
**Purpose**: Container for complete retrieval results
**Contents**:
- `primary_chunks`: Direct search results (what the query matched)
- `referenced_chunks`: Auto-discovered dependencies (what you need to understand the primary results)
- `total_tokens`: Estimated token count for LLM context planning
- `retrieval_stats`: Performance metrics for monitoring

## Reference Expansion Algorithm Explained

### The Breadth-First Strategy
**Why breadth-first?** Ensures we get the most immediately relevant dependencies before going deeper into the reference graph.

**Step-by-step process**:

1. **Initialize**: Start with primary chunks, mark them as visited
2. **Collect Level 1**: Extract all `ref_ids` from primary chunks, add to expansion queue
3. **Process Level 1**: Fetch chunks by ID, add to results, mark as visited
4. **Collect Level 2**: Extract `ref_ids` from Level 1 chunks, add to queue
5. **Continue**: Repeat until max_depth reached or max_total chunks collected

**Circular Protection**:
- `visited_ids` set prevents processing the same chunk twice
- Depth tracking prevents infinite recursion
- Early termination when hitting chunk limits

### Example Reference Expansion

**Starting point**: Primary chunk `api.json:paths/users/post`

**Level 1 expansion** (from primary chunk's ref_ids):
- `api.json:components/schemas/User` (User schema)
- `api.json:paths/users/post:errors` (Error responses)

**Level 2 expansion** (from Level 1 chunks' ref_ids):
- `api.json:components/schemas/Address` (from User schema)
- `api.json:components/schemas/ValidationError` (from error responses)

**Level 3 expansion** (from Level 2 chunks' ref_ids):
- `api.json:components/schemas/Country` (from Address schema)

**Result**: Complete context with 6 chunks covering the entire dependency tree needed to understand "how to create a user"

## Query Analysis and Optimization

### Intent Recognition
**Purpose**: Optimize retrieval strategy based on what the user is really asking

**Query patterns detected**:
- **Endpoint queries**: "how to create", "list users", "update profile" 
  → Focus search on operation chunks, include error responses
- **Schema queries**: "what fields", "user object structure", "data model"
  → Focus search on component chunks, include examples
- **Error queries**: "error codes", "failed request", "exception handling"
  → Focus search on error response chunks

### Metadata Filtering
**When applied**: Based on query intent analysis
**How it works**: Adds `filters` parameter to initial Vector Store Manager search
**Fallback strategy**: If filtered search returns < 3 results, retry without filters

**Example**: Query "user schema fields" triggers:
```python
filters = {"type": "component"}  # Focus on schema components
```

## Performance Considerations

### Token Estimation
**Method**: Uses `get_token_count()` from embedding_utils with tiktoken
**Purpose**: Ensures context fits within LLM limits (4K default)
**Applied to**: Combined document text from all chunks in final context

### Circular Reference Detection
**Problem**: Schema A references Schema B, which references Schema A
**Solution**: `visited_ids` set tracks all processed chunks
**Behavior**: Skip already-visited chunks, log occurrence for monitoring

### Batch Optimization (Future)
**Current**: Individual `get_by_ids([single_id])` calls
**Planned**: Collect all needed IDs for current depth level, single `get_by_ids([id1, id2, ...])` call
**Tradeoff**: More complex logic vs. fewer database round trips

## Context Size Management

### Two-Limit System
1. **Chunk limit**: `max_total_chunks` (default: 15) - prevents too many pieces
2. **Token limit**: `CONTEXT_TOKEN_LIMIT` (default: 4000) - prevents context overflow

### Prioritization Strategy
**When hitting limits**:
1. Always include all primary chunks (direct search results)
2. Fill remaining slots with highest-relevance referenced chunks
3. Apply 60/40 split: reserve 60% slots for primary, 40% for references
4. If primary chunks exceed 60%, reduce reference allocation

### Early Termination
**Trigger**: When chunk or token limits approached
**Behavior**: Stop reference expansion, return partial but valid context
**Logging**: Record what limits were hit for optimization

## Error Handling and Edge Cases

### Missing References
**Scenario**: `ref_ids` points to chunk that doesn't exist in vector store
**Behavior**: Log warning, continue with available chunks
**Impact**: Graceful degradation - return partial but valid context

### Empty Search Results
**Scenario**: Query returns no matching chunks
**Behavior**: Return empty but well-formed `KnowledgeContext`
**Structure**: All lists empty, zero counts, but valid object structure

### Malformed Metadata
**Scenario**: Chunk missing `ref_ids` field or has invalid format
**Behavior**: Treat as chunk with no references, continue processing
**Logging**: Debug-level log for investigation

### Timeout Handling
**Scenario**: Reference expansion takes too long
**Behavior**: Return partial results with timeout flag in stats
**Configuration**: `RETRIEVAL_TIMEOUT_MS` (default: 5000ms)

## Integration with Other Components

### Upstream: Vector Store Manager
**Uses**:
- `search(query, limit, filters)` for semantic search
- `get_by_ids([ids])` for reference expansion
**Expects**: Standard chunk format with id, document, metadata fields

### Downstream: MCP Server
**Provides**: Complete `KnowledgeContext` object
**Contract**: All necessary information for LLM to answer query completely
**No LLM integration**: Knowledge Retriever is LLM-agnostic

### Data Dependencies: OpenAPI Processor
**Relies on**: Properly populated `ref_ids` in chunk metadata
**Expects**: File-path-based IDs (`filename:element_path`)
**Assumes**: Bidirectional references maintained by chunk assembler

## Configuration and Behavioral Control

### Core Retrieval Settings
- `RETRIEVAL_MAX_PRIMARY_RESULTS=5`: How many initial search results
- `RETRIEVAL_MAX_TOTAL_CHUNKS=15`: Total context size limit  
- `RETRIEVAL_MAX_DEPTH=3`: How deep to follow reference chains
- `CONTEXT_TOKEN_LIMIT=4000`: Token budget for final context

### Query Analysis Settings
- `ENABLE_QUERY_ANALYSIS=true`: Use intent recognition for optimization
- `CONTEXT_PRIORITIZE_PRIMARY=true`: Always include primary results first

### Performance Settings
- `RETRIEVAL_TIMEOUT_MS=5000`: Maximum time for complete retrieval
- `LOG_RETRIEVAL_STATS=true`: Detailed performance logging
- `BATCH_REFERENCE_LOOKUP=true`: Enable batch optimization (future)

## Success Metrics and Validation

### Retrieval Quality
- **Completeness**: >90% of queries return all necessary dependencies
- **Precision**: Retrieved chunks are relevant to query
- **Consistency**: Same query returns same essential chunks

### Performance Targets
- **Speed**: <200ms total retrieval time (search + expansion + assembly)
- **Memory**: <512MB during reference expansion
- **Scalability**: Performance stable with 200+ API specifications

### Error Resilience
- **Reference Resolution**: 100% success rate for valid `ref_ids`
- **Circular Reference Protection**: Never infinite loop
- **Graceful Degradation**: Always return valid context, even with errors

This approach ensures that when someone asks "How do I create a user?", they get not just the endpoint definition, but everything needed to actually implement it correctly.