# Intelligent Chunk Filtering Component Specification

## Component Purpose

Enhance the Research Agent's search capabilities by implementing LLM-based relevance filtering to reduce context explosion and improve result quality. This component addresses the core issue where vector similarity search returns many loosely relevant chunks, overwhelming the LLM with irrelevant context.

## Problem Statement

### Current Issues
The existing `searchChunks` implementation suffers from context quality problems:

1. **Over-Retrieval**: Vector search returns 25 chunks by default, many with low actual relevance
2. **Context Explosion**: Reference expansion in `getChunks` amplifies irrelevant content
3. **Query Drift**: Expanded chunks often relate to the original chunks but not the user's actual question
4. **Fixed Limits**: Static limits don't adapt to query complexity or result quality

### Real-World Impact
```
User Query: "How do I authenticate with sponsored-display API?"
Current Flow:
  ├── Vector search finds 25 chunks about authentication, errors, schemas
  ├── Many chunks about rate limits, field validation, error codes
  ├── Reference expansion adds 50+ more chunks
  └── Result: 75 chunks, 60% irrelevant to authentication

Desired Flow:
  ├── Vector search finds 40 chunks (oversample)
  ├── LLM filters to 8 highly relevant authentication chunks
  ├── Reference expansion works with focused, relevant base
  └── Result: 25 chunks, 90% relevant to authentication
```

## Solution Architecture

### High-Level Flow
```
User Query → searchChunks() → Vector Search (oversample) → LLM Intelligence Engine → Smart Results
                                    ↓                            ↓
                               Full chunk content      Relevance + Expansion Decisions
                                    ↓                            ↓
                               Fallback: Trim to max_chunks if LLM fails
```

### Core Components

1. **Oversample Strategy**: Search for more chunks than requested to give LLM filtering options
2. **LLM Intelligence Engine**: Use existing Haiku model for relevance assessment AND expansion decisions
3. **Intelligent Context Management**: LLM decides which chunks to keep, expand shallow, or expand deep
4. **Automatic Reference Expansion**: Execute LLM expansion decisions with appropriate depths
5. **Graceful Fallback**: Return top N chunks if LLM filtering fails
6. **Performance Monitoring**: Track filtering effectiveness and performance

## Technical Implementation

### 1. Enhanced searchChunks Function

**File**: `src/research_agent/tools.py`

```python
async def searchChunks(
    vector_store: VectorStoreManager,
    api_context: str,
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False,
    rerank: bool = True,  # NEW: Enable LLM re-ranking
    oversample_multiplier: float = 1.6,  # NEW: Configurable oversample rate
) -> SearchResults:
    """
    Search for relevant chunks with optional LLM-based relevance filtering.
    
    Args:
        vector_store: VectorStoreManager instance for search operations
        api_context: Available API files context string
        query: Natural language search query
        max_chunks: Maximum chunks to return (default: 25)
        file_filter: Optional file pattern (e.g., "sponsored-display", "dsp")
        include_references: Whether to include ref_ids in response
        rerank: Whether to apply LLM-based relevance filtering (default: True)
        oversample_multiplier: How many extra chunks to search before filtering (default: 1.6x)
    
    Returns:
        SearchResults with filtered chunk summaries and metadata
    """
```

### 2. LLM Relevance Filtering Implementation

```python
async def _filter_and_expand_chunks(
    chunks: List[ChunkSummary],
    query: str,
    target_count: int,
    model: ChatBedrockConverse,
    vector_store: VectorStoreManager
) -> Tuple[List[FullChunk], Dict[str, Any]]:
    """
    Use LLM to filter chunks and make intelligent expansion decisions.
    
    Args:
        chunks: List of chunk summaries from vector search
        query: Original user query
        target_count: Desired number of base chunks (before expansion)
        model: LLM model for relevance assessment and expansion decisions
        vector_store: For executing expansion decisions
        
    Returns:
        Tuple of (processed_chunks, processing_stats)
    """
    start_time = time.time()
    
    # Prepare chunk information for LLM with full content and references
    chunk_info = []
    for i, chunk in enumerate(chunks):
        chunk_info.append({
            "index": i,
            "id": chunk.chunk_id,
            "title": chunk.title,
            "content": chunk.content_preview,  # Now contains full content
            "type": chunk.chunk_type,
            "file": chunk.file_name,
            "ref_ids": chunk.ref_ids,  # Available references for expansion decisions
            "relevance_score": chunk.relevance_score
        })
    
    # LLM prompt for relevance and expansion decisions (to be experimentally tuned)
    prompt = _build_intelligence_prompt(query, chunk_info, target_count)
    
    try:
        # Call LLM for relevance assessment and expansion decisions
        response = await model.ainvoke([{"role": "user", "content": prompt}])
        
        # Parse LLM response to get chunk decisions by ID
        chunk_decisions = _parse_relevance_response(response.content)
        
        # Create chunk lookup by ID
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Process chunks based on LLM decisions
        processed_chunks = []
        for chunk_id, action in chunk_decisions.items():
            if chunk_id not in chunk_lookup or action == "DISCARD":
                continue
                
            chunk = chunk_lookup[chunk_id]
            
            if action == "KEEP":
                # Add chunk without expansion
                full_chunk = await _convert_to_full_chunk(chunk, vector_store, expand_depth=0)
                processed_chunks.append(full_chunk)
            elif action.startswith("EXPAND_"):
                # Extract expansion depth and expand
                try:
                    depth = int(action.split("_")[1])
                    expanded_result = await getChunks(
                        vector_store=vector_store,
                        chunk_ids=[chunk.chunk_id],
                        expand_depth=depth,
                        max_total_chunks=50  # Reasonable limit per chunk
                    )
                    processed_chunks.extend(expanded_result.requested_chunks)
                    processed_chunks.extend(expanded_result.expanded_chunks)
                except (ValueError, IndexError):
                    # Fallback to no expansion if parsing fails
                    full_chunk = await _convert_to_full_chunk(chunk, vector_store, expand_depth=0)
                    processed_chunks.append(full_chunk)
        
        filtering_time = (time.time() - start_time) * 1000
        
        stats = {
            "llm_intelligence_enabled": True,
            "original_count": len(chunks),
            "processed_count": len(processed_chunks),
            "decisions": chunk_decisions,
            "processing_time_ms": filtering_time,
            "fallback_used": False
        }
        
        return processed_chunks, stats
        
    except Exception as e:
        # Fallback: return top chunks by vector similarity without expansion
        logger.warning(f"LLM intelligence failed: {e}. Using fallback.")
        fallback_chunks = []
        for chunk in chunks[:target_count]:
            full_chunk = await _convert_to_full_chunk(chunk, vector_store, expand_depth=0)
            fallback_chunks.append(full_chunk)
        
        stats = {
            "llm_intelligence_enabled": True,
            "original_count": len(chunks),
            "processed_count": len(fallback_chunks),
            "decisions": {},
            "processing_time_ms": 0,
            "fallback_used": True,
            "error": str(e)
        }
        
        return fallback_chunks, stats
```

### 3. Enhanced LangChain Tool Wrapper

**File**: `src/research_agent/agent_tools.py`

```python
@tool
async def search_chunks_tool(
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False,
    rerank: bool = True,  # NEW: Expose rerank parameter
) -> dict:
    """Search API documentation chunks with optional LLM-based relevance filtering.

    Use this tool to find relevant documentation chunks. The rerank parameter enables
    intelligent filtering using LLM assessment of chunk relevance to your query.
    """
    logger.info(f"search_chunks_tool called: query='{query}', max_chunks={max_chunks}, "
                f"file_filter={file_filter}, rerank={rerank}")

    resources = get_shared_resources()
    api_context = generate_api_context()

    result = await searchChunks(
        vector_store=resources.vector_store,
        api_context=api_context,
        query=query,
        max_chunks=max_chunks,
        file_filter=file_filter,
        include_references=include_references,
        rerank=rerank,
    )

    logger.info(f"search_chunks_tool result: {result.total_found} chunks found in {result.search_time_ms:.1f}ms")
    
    # Log filtering stats if available
    if hasattr(result, 'filtering_stats'):
        stats = result.filtering_stats
        logger.info(f"LLM filtering: {stats['original_count']} → {stats['filtered_count']} chunks "
                   f"({stats['reduction_ratio']:.1%} reduction), "
                   f"fallback_used={stats['fallback_used']}")

    # Convert to dict for LLM consumption (include filtering stats)
    result_dict = {
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content_preview": chunk.content_preview,
                "chunk_type": chunk.chunk_type,
                "file_name": chunk.file_name,
                "ref_ids": chunk.ref_ids,
                "relevance_score": chunk.relevance_score,
            }
            for chunk in result.chunks
        ],
        "total_found": result.total_found,
        "search_time_ms": result.search_time_ms,
        "files_searched": result.files_searched,
        "api_context": result.api_context,
    }
    
    # Include filtering stats if available
    if hasattr(result, 'filtering_stats'):
        result_dict["filtering_stats"] = result.filtering_stats
    
    return result_dict
```

## Prompt Engineering Framework

### Initial Prompt Structure
The LLM relevance filtering prompt needs experimentation to optimize results. Initial structure:

```python
def _build_intelligence_prompt(query: str, chunk_info: List[Dict], target_count: int) -> str:
    """Build prompt for LLM relevance assessment and expansion decisions (experimental)."""
    
    # This prompt structure needs experimental tuning
    prompt = f"""You are an expert API documentation analyst. Evaluate chunks for relevance and decide appropriate expansion strategy.

User Query: "{query}"

For each relevant chunk, decide the best action to provide comprehensive but focused context.

Chunks to evaluate:
"""
    
    for chunk in chunk_info:
        prompt += f"""
Chunk ID: {chunk['id']}
Title: {chunk['title']}
Type: {chunk['type']}
File: {chunk['file']}
References: {chunk['ref_ids']}
Vector Score: {chunk['relevance_score']:.3f}
Content:
{chunk['content']}

---
"""
    
    prompt += f"""
Instructions:
1. Evaluate each chunk for relevance to the user's query
2. For relevant chunks, decide the appropriate action based on content and references:
   - KEEP: Chunk is complete and directly answers the query
   - EXPAND_1: Needs basic context from immediate references
   - EXPAND_3: Needs moderate expansion for schemas or related concepts
   - EXPAND_5: Needs deep expansion for complex nested structures
   - DISCARD: Not relevant to the query
3. Consider chunk type and references when deciding expansion depth
4. Aim for comprehensive but focused results

Response format: One decision per line
chunk_id -> action

Example:
sd-api:CreateCampaign -> EXPAND_3
dsp-api:GetAudiences -> KEEP
sd-api:ErrorCodes -> DISCARD

Note: Any chunk not mentioned is automatically discarded
"""
    
    return prompt

def _parse_relevance_response(response: str) -> Dict[str, str]:
    """Parse LLM response to extract chunk decisions by ID."""
    try:
        # Parse line-by-line "chunk_id -> action" format (handles spaces around ->)
        decisions = {}
        for line in response.strip().split('\n'):
            line = line.strip()
            if '->' in line and line:
                # Split on arrow separator, handling optional spaces
                parts = line.split('->', 1)
                if len(parts) == 2:
                    chunk_id, action = parts
                    decisions[chunk_id.strip()] = action.strip()
        return decisions
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {response}. Error: {e}")
        return {}  # Return empty dict to trigger fallback
```

### Experimental Tuning Process
1. **Test Multiple Prompt Variants**: Different instruction styles, context levels, output formats
2. **Evaluate Against Test Queries**: Use known good/bad chunk pairs for validation
3. **Measure Performance**: Relevance improvement vs. processing time trade-offs
4. **A/B Testing**: Compare rerank=True vs rerank=False results

## Configuration System

### Environment Variables
```bash
# Intelligent Chunk Filtering Configuration
CHUNK_FILTERING_ENABLED=true                    # Global enable/disable
CHUNK_FILTERING_OVERSAMPLE_MULTIPLIER=1.6       # How much to oversample (1.0 = no oversampling)
CHUNK_FILTERING_MIN_RELEVANCE_THRESHOLD=0.3     # Minimum vector score to consider
CHUNK_FILTERING_LLM_TIMEOUT_MS=5000            # LLM call timeout before fallback
CHUNK_FILTERING_MAX_PROMPT_TOKENS=2000         # Limit prompt size for performance
```

### Configuration Integration
```python
# In tools.py
async def searchChunks(...):
    # Get configuration from environment or config object
    config = get_config()
    
    # Use configurable defaults
    if rerank is None:
        rerank = config.get('chunk_filtering_enabled', True)
    
    if oversample_multiplier is None:
        oversample_multiplier = config.get('chunk_filtering_oversample_multiplier', 1.6)
    
    # Calculate oversample count
    if rerank:
        search_limit = int(max_chunks * oversample_multiplier)
    else:
        search_limit = max_chunks
```

## Performance Monitoring

### Metrics to Track
1. **Filtering Effectiveness**:
   - Original chunk count vs. filtered count
   - Reduction ratio (percentage of chunks filtered out)
   - Fallback usage frequency

2. **Performance Impact**:
   - LLM filtering time (separate from vector search time)
   - Total query response time increase
   - Token usage for filtering prompts

3. **Quality Indicators**:
   - User satisfaction with filtered results
   - Downstream reference expansion efficiency
   - Agent tool selection accuracy

### Logging Implementation
```python
# Enhanced logging in searchChunks
logger.info(f"Chunk filtering results: "
           f"original={stats['original_count']}, "
           f"filtered={stats['filtered_count']}, "
           f"reduction={stats['reduction_ratio']:.1%}, "
           f"time={stats['filtering_time_ms']:.1f}ms, "
           f"fallback={stats['fallback_used']}")

# Integration with existing metrics
if hasattr(result, 'filtering_stats'):
    # Add filtering stats to SearchResults for upstream logging
    result.filtering_stats = stats
```

## Testing Strategy

### 1. Unit Tests
```python
# Test LLM filtering functionality
async def test_llm_filtering_basic():
    """Test basic LLM filtering with mock chunks"""
    
async def test_llm_filtering_fallback():
    """Test fallback when LLM filtering fails"""
    
async def test_oversample_calculation():
    """Test oversample multiplier calculation"""
```

### 2. Integration Tests
```python
# Test with real data scenarios
async def test_authentication_query_filtering():
    """Test filtering for authentication-related queries"""
    
async def test_error_handling_query_filtering():
    """Test filtering for error handling queries"""
    
async def test_complex_schema_query_filtering():
    """Test filtering for complex schema navigation queries"""
```

### 3. Before/After Comparison Framework
```python
async def compare_filtering_results(test_queries: List[str]):
    """Compare results with and without filtering"""
    for query in test_queries:
        # Test without filtering
        result_baseline = await searchChunks(query, rerank=False)
        
        # Test with filtering  
        result_filtered = await searchChunks(query, rerank=True)
        
        # Analyze differences
        analyze_filtering_impact(result_baseline, result_filtered, query)
```

### 4. Performance Benchmarks
- Measure response time impact across different query types
- Test with various oversample multipliers (1.2x, 1.6x, 2.0x)
- Validate memory usage with large chunk sets

## Integration with Research Agent

### Agent Tool Usage
The Research Agent will automatically benefit from improved search results:

```python
# Agent uses existing search_chunks_tool interface
# No changes needed to agent.py or prompt
agent = create_react_agent(
    model=model,
    tools=[search_chunks_tool, get_chunks_tool],  # Same interface
    prompt=existing_prompt  # No prompt changes needed
)
```

### Backward Compatibility
- Default `rerank=True` provides improved results by default
- `rerank=False` preserves existing behavior for comparison
- All existing function signatures remain compatible
- No changes required to MCP server integration

## Implementation Plan

### Phase 1: Core Filtering Implementation
1. **Add LLM filtering logic** to `searchChunks()` function
2. **Implement oversample calculation** and configuration loading
3. **Create initial relevance prompt** with basic structure
4. **Add fallback mechanism** for LLM failures
5. **Update SearchResults** data class to include filtering stats

### Phase 2: Tool Integration
1. **Update `search_chunks_tool()`** to expose rerank parameter
2. **Add configuration system** for filtering parameters
3. **Implement comprehensive logging** for filtering metrics
4. **Create unit tests** for filtering functionality

### Phase 3: Experimental Tuning
1. **Design prompt experiments** with multiple variants
2. **Create evaluation dataset** with known good/bad examples
3. **Run A/B testing framework** for prompt optimization
4. **Measure performance impact** and optimize if needed

### Phase 4: Production Deployment
1. **Integration testing** with real research agent workflows
2. **Performance validation** under production load
3. **Monitoring setup** for filtering effectiveness
4. **Documentation updates** for new parameters

## Success Criteria

### Functionality
- ✅ LLM filtering reduces irrelevant chunks by 30-50%
- ✅ Fallback mechanism works reliably when LLM fails
- ✅ Configuration system allows easy tuning
- ✅ All existing functionality preserved

### Performance
- ✅ LLM filtering adds <3 seconds to query response time
- ✅ Memory usage remains within acceptable limits
- ✅ Filtering effectiveness >70% (measured by human evaluation)

### Integration
- ✅ No changes required to existing agent prompts or tools
- ✅ Backward compatibility with rerank=False option
- ✅ Comprehensive logging and monitoring in place
- ✅ Easy deployment without breaking existing functionality

## Future Enhancements

### Short-term
1. **Prompt optimization** based on experimental results
2. **Caching layer** for repeated relevance assessments
3. **Adaptive oversample multipliers** based on query complexity

### Long-term
1. **Learned relevance patterns** from agent usage
2. **Multi-criteria filtering** (relevance + quality + completeness)
3. **Cross-query relevance** for conversation context

## Risk Mitigation

### Technical Risks
- **LLM Reliability**: Fallback to vector similarity ensures continuity
- **Performance Impact**: Configurable timeouts and caching minimize delays
- **Cost Increase**: Monitor token usage and implement limits if needed

### Deployment Risks
- **Backward Compatibility**: Preserved through default parameters
- **Configuration Errors**: Sensible defaults and validation
- **Monitoring Gaps**: Comprehensive logging from day one

This specification provides a complete roadmap for implementing intelligent chunk filtering while maintaining simplicity and reliability. The experimental framework ensures we can tune the system for optimal results while the fallback mechanisms guarantee robustness.