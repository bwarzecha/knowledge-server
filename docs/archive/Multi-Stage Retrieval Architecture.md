# Multi-Stage Retrieval Architecture Design

## Overview

This document outlines the architecture for intelligent, context-aware retrieval that replaces static chunk enhancement with dynamic, on-demand context resolution. The system uses a three-stage pipeline with smart API pre-filtering to handle massive scale (200+ API specifications) while providing complete, accurate context.

## Architecture Principles

### 1. Dynamic Over Static
- **No pre-enhancement hallucination** - Only retrieve factual data that exists
- **On-demand context resolution** - Get exactly what each query needs
- **Intelligent traversal** - Follow actual ref_ids in dependency graph

### 2. Scalable Filtering
- **File-based isolation** - Each API spec is a separate searchable domain
- **Progressive filtering** - Narrow search space at each stage
- **Metadata-driven search** - Use ChromaDB filtering for performance

### 3. Complete Context
- **Automatic ref resolution** - Follow schema dependencies to completion
- **Circular reference protection** - Prevent infinite traversal loops
- **Bounded depth** - Limit traversal to prevent explosion

## Three-Stage Pipeline

### Stage 1: API Selection LLM
**Purpose**: Identify which APIs are relevant to the user's query

**Input**: 
- User query
- Compact API catalog (file-based)

**Context Structure**:
```python
api_catalog = [
    {
        "file_id": "amazon_advertising_v3", 
        "name": "Amazon Advertising API", 
        "description": "Sponsored Products, Brands, Display campaigns",
        "domains": ["advertising", "campaigns", "keywords", "targeting"]
    },
    {
        "file_id": "stripe_v2023", 
        "name": "Stripe API", 
        "description": "Payments, billing, subscriptions, customers",
        "domains": ["payments", "billing", "finance", "subscriptions"]
    },
    # ... 200+ API specifications
]
```

**Process**:
1. LLM analyzes user query for business domain keywords
2. Matches query intent to API descriptions and domains
3. Selects 2-3 most relevant file_ids

**Output**: 
```python
{
    "selected_apis": ["amazon_advertising_v3", "facebook_ads_v17"],
    "reasoning": "Query about campaign creation relates to advertising APIs"
}
```

**Example**:
- Query: "How do I create a campaign?"
- Output: `["amazon_advertising_v3", "facebook_ads_v17"]`

### Stage 2: Query Expansion LLM
**Purpose**: Expand user query with specific API terminology from selected APIs

**Input**:
- Original user query  
- Detailed API indexes for selected file_ids only

**Context Structure** (per selected API):
```python
api_index = {
    "file_id": "amazon_advertising_v3",
    "endpoints": [
        {
            "path": "/campaigns", 
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "operations": ["listCampaigns", "createCampaigns", "updateCampaigns", "archiveCampaign"]
        },
        {
            "path": "/ad-groups",
            "methods": ["GET", "POST", "PUT"],
            "operations": ["listAdGroups", "createAdGroups", "updateAdGroups"]
        }
    ],
    "schemas": [
        {
            "name": "Campaign", 
            "type": "response", 
            "used_by": ["listCampaigns", "getCampaign"]
        },
        {
            "name": "CreateCampaign", 
            "type": "request", 
            "used_by": ["createCampaigns"]
        }
    ],
    "domains": [
        {
            "name": "Campaigns", 
            "endpoints": ["/campaigns"], 
            "schemas": ["Campaign", "CreateCampaign", "UpdateCampaign"]
        }
    ]
}
```

**Process**:
1. LLM analyzes user intent against API-specific terminology
2. Adds relevant endpoint names, schema names, and operation IDs
3. Preserves original user language while adding technical terms

**Output**:
```python
{
    "expanded_query": "create campaign POST /campaigns CreateCampaign schema sponsored products advertising",
    "original_terms": ["create", "campaign"],
    "added_terms": ["POST", "/campaigns", "CreateCampaign", "sponsored products"]
}
```

**Example**:
- Input: "How do I create a campaign?" + amazon-advertising index
- Output: "create campaign POST /campaigns CreateCampaign schema sponsored products operation"

### Stage 3: Filtered Search + Context Resolution
**Purpose**: Find relevant chunks and automatically resolve all dependencies

**Input**:
- Expanded query from Stage 2
- Selected file_ids from Stage 1

**Process**:

#### 3a. Filtered Vector Search
```python
# ChromaDB search with file-based filtering
results = collection.query(
    query_texts=[expanded_query],
    where={"source_file": {"$in": selected_file_ids}},
    n_results=10
)
```

#### 3b. Automatic Ref Resolution
```python
def resolve_all_refs(chunk_id: str, max_depth: int = 3) -> List[Chunk]:
    """Recursively resolve all ref_ids starting from chunk_id"""
    visited = set()
    result = []
    
    def traverse(current_id: str, depth: int):
        if depth > max_depth or current_id in visited:
            return
        
        visited.add(current_id)
        chunk = get_chunk_by_id(current_id)
        
        if chunk:
            result.append(chunk)
            
            # Follow all references
            for ref_id in chunk.metadata.get('ref_ids', []):
                traverse(ref_id, depth + 1)
    
    traverse(chunk_id, 0)
    return result
```

**Output**:
```python
{
    "primary_chunks": [top_search_results],
    "resolved_context": [all_referenced_schemas_and_dependencies],
    "total_chunks": 15,
    "traversal_depth": 3
}
```

## Implementation Details

### API Catalog Generation
```python
def generate_api_catalog(specs_directory: str) -> List[Dict]:
    """Generate compact API catalog from all spec files"""
    catalog = []
    
    for spec_file in glob.glob(f"{specs_directory}/*.{json,yaml}"):
        spec = load_openapi_spec(spec_file)
        
        # Extract high-level information
        file_id = Path(spec_file).stem
        name = spec.get('info', {}).get('title', file_id)
        description = spec.get('info', {}).get('description', '')
        
        # Extract business domains from tags/paths
        domains = extract_business_domains(spec)
        
        catalog.append({
            "file_id": file_id,
            "name": name,
            "description": description[:200],  # Truncate for context
            "domains": domains
        })
    
    return catalog
```

### Detailed Index Generation  
```python
def generate_detailed_index(spec_file: str) -> Dict:
    """Generate detailed API index for query expansion"""
    spec = load_openapi_spec(spec_file)
    
    endpoints = []
    schemas = []
    
    # Extract all endpoints
    for path, path_item in spec.get('paths', {}).items():
        methods = list(path_item.keys())
        operations = [op.get('operationId') for op in path_item.values() 
                     if op.get('operationId')]
        
        endpoints.append({
            "path": path,
            "methods": methods,
            "operations": operations
        })
    
    # Extract all schemas
    for schema_name, schema_def in spec.get('components', {}).get('schemas', {}).items():
        # Find which operations use this schema
        used_by = find_operations_using_schema(spec, schema_name)
        
        schemas.append({
            "name": schema_name,
            "type": infer_schema_type(schema_def),
            "used_by": used_by
        })
    
    return {
        "file_id": Path(spec_file).stem,
        "endpoints": endpoints,
        "schemas": schemas,
        "domains": group_by_business_domain(endpoints, schemas)
    }
```

### ChromaDB Integration
```python
def setup_filtered_search(chunks: List[Chunk]) -> chromadb.Collection:
    """Setup ChromaDB collection with file-based filtering"""
    
    # Ensure each chunk has source_file metadata
    for chunk in chunks:
        chunk.metadata['source_file'] = extract_file_id(chunk.id)
    
    collection = client.create_collection(
        name="api_knowledge_filtered",
        embedding_function=embedding_function
    )
    
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.document for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks]
    )
    
    return collection
```

## Performance Characteristics

### Scalability
- **API Catalog Size**: O(n) where n = number of API files
- **Context Window Usage**: O(1) - only selected APIs loaded
- **Search Performance**: O(log n) with file filtering
- **Memory Usage**: Constant per query regardless of total APIs

### Accuracy Benefits
- **Precision**: Search only in relevant API domains
- **Completeness**: Automatic dependency resolution ensures nothing is missed
- **Consistency**: Follows actual ref_ids, no hallucination

### Cost Analysis
- **3 LLM calls per query** (vs 1 call with massive context)
- **Smaller context windows** (lower token costs)
- **Better cache hit rates** (repeated API selections)

## Error Handling

### Circular Reference Protection
```python
def safe_traverse(current_id: str, visited: Set[str], max_depth: int) -> bool:
    """Prevent infinite loops in schema references"""
    if current_id in visited:
        logger.warning(f"Circular reference detected: {current_id}")
        return False
    
    if len(visited) >= max_depth:
        logger.warning(f"Max depth {max_depth} reached")
        return False
    
    return True
```

### Missing Reference Handling
```python
def resolve_ref_safely(ref_id: str) -> Optional[Chunk]:
    """Handle missing schema references gracefully"""
    try:
        return get_chunk_by_id(ref_id)
    except ChunkNotFoundError:
        logger.warning(f"Referenced chunk not found: {ref_id}")
        return None
```

### API Selection Fallback
```python
def fallback_api_selection(query: str) -> List[str]:
    """Fallback when LLM fails to select APIs"""
    # Use keyword matching as backup
    keywords = extract_keywords(query)
    return match_apis_by_keywords(keywords, api_catalog)
```

## Future Enhancements

### Intelligent Caching
- Cache API selections for similar queries
- Cache resolved context subgraphs
- LRU eviction for memory management

### Adaptive Depth Control
- Adjust max traversal depth based on query complexity
- Stop early when sufficient context is gathered
- Learn optimal depths from user feedback

### Cross-API Resolution
- Resolve references that span multiple API specifications
- Handle shared schema libraries
- Support federated API ecosystems

## Success Metrics

### Accuracy Metrics
- **Context Completeness**: % of queries with all necessary schemas resolved
- **Precision**: % of selected APIs that contain relevant results
- **Traversal Efficiency**: Average refs resolved per query

### Performance Metrics  
- **Query Latency**: Total time for 3-stage pipeline
- **Context Size**: Average tokens in final resolved context
- **Cache Hit Rate**: % of API selections served from cache

### Scale Metrics
- **API Support**: Number of APIs handled without performance degradation
- **Concurrent Queries**: Simultaneous queries supported
- **Memory Usage**: Peak memory during large context resolution

## Conclusion

This multi-stage retrieval architecture transforms the knowledge server from a simple vector search into an intelligent API navigation system. By combining smart pre-filtering with automatic dependency resolution, it provides complete, accurate context while scaling to hundreds of API specifications.

The key innovation is moving complexity from static pre-processing (which leads to hallucination) to dynamic retrieval (which follows factual relationships). This enables both massive scale and high accuracy - the two requirements that seemed impossible to achieve simultaneously with previous approaches.