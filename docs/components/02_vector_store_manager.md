# Vector Store Manager Component Specification

## Component Purpose

Manage ChromaDB operations including document storage, embedding generation, and filtered semantic search. This component encapsulates all vector database interactions and provides the foundation for the intelligent retrieval system validated in prototype testing.

## Core Responsibilities

1. **ChromaDB Collection Management**: Initialize and configure ChromaDB collections with proper settings
2. **Embedding Model Integration**: Configure and use Qwen3-0.6B (or configurable alternatives) with MPS acceleration
3. **Chunk Storage**: Store chunks with metadata and generate embeddings efficiently
4. **Semantic Search**: Perform vector similarity search with metadata filtering
5. **ID-Based Retrieval**: Provide fast lookup of chunks by ID for reference resolution
6. **Collection Persistence**: Handle data persistence and collection lifecycle

## Input/Output Contracts

### Input Chunks (from OpenAPI Processor)
```python
{
    "id": str,           # File-path-based ID: "{filename}:{natural_name}"
    "document": str,     # Human-readable text for embedding
    "metadata": dict     # All metadata from OpenAPI Processor
}
```

### Search Query Interface
```python
def search(
    query: str,
    limit: int = 5,
    filters: Optional[Dict] = None,
    include_metadata: bool = True
) -> List[Dict]:
    """Semantic search with optional filtering"""

def get_by_ids(
    ids: List[str],
    include_metadata: bool = True
) -> List[Dict]:
    """Direct retrieval by chunk IDs"""
```

### Output Format
```python
{
    "id": str,
    "document": str,
    "metadata": dict,
    "distance": float,    # Similarity score (search only)
    "rank": int          # Result ranking (search only)
}
```

## Key Implementation Details

### ChromaDB Configuration
**Validated Settings**: Based on prototype testing that achieved 60% relevance with 100ms query time.

```python
# Collection setup with validated configuration
collection = client.create_collection(
    name=config.CHROMA_COLLECTION_NAME,
    embedding_function=Qwen3EmbeddingFunction(
        model_name=config.EMBEDDING_MODEL,
        device=config.EMBEDDING_DEVICE
    ),
    metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
)
```

### Custom Embedding Function
**Critical Implementation**: Proper Qwen3 usage requires query-specific encoding.

```python
class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, device: str = "mps"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.is_query = False  # Track context for proper encoding
    
    def __call__(self, input: Documents) -> Embeddings:
        # Use different encoding for queries vs documents
        if self.is_query:
            embeddings = self.model.encode(list(input), prompt_name="query")
        else:
            embeddings = self.model.encode(list(input))
        return embeddings.tolist()
    
    def encode_query(self, query: str) -> List[float]:
        """Encode search query with proper prompt"""
        self.is_query = True
        embedding = self.model.encode([query], prompt_name="query")[0]
        self.is_query = False
        return embedding.tolist()
```

### Metadata Handling
**ChromaDB Constraint**: Metadata values must be scalars (string, int, float, bool), not lists.

```python
def prepare_metadata_for_chromadb(metadata: Dict) -> Dict:
    """Convert complex metadata to ChromaDB-compatible format"""
    compatible = {}
    
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convert lists to comma-separated strings
            compatible[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (dict, object)):
            # Convert complex objects to JSON strings
            compatible[key] = json.dumps(value)
        else:
            # Keep scalars as-is
            compatible[key] = value
    
    return compatible

def restore_metadata_from_chromadb(metadata: Dict) -> Dict:
    """Convert ChromaDB metadata back to original format"""
    restored = {}
    
    # Define which fields should be lists
    list_fields = {"ref_ids", "tags", "used_by_operations", "properties"}
    
    for key, value in metadata.items():
        if key in list_fields and isinstance(value, str):
            # Convert comma-separated strings back to lists
            restored[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            restored[key] = value
    
    return restored
```

### Efficient Search Implementation
**Performance Target**: <100ms query time based on prototype validation.

```python
def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """Optimized semantic search with filtering"""
    
    # Encode query with proper prompt
    query_embedding = self.embedding_function.encode_query(query)
    
    # Build ChromaDB where clause from filters
    where_clause = self._build_where_clause(filters) if filters else None
    
    # Perform vector search
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results with restored metadata
    formatted_results = []
    for i in range(len(results["ids"][0])):
        result = {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": self.restore_metadata_from_chromadb(results["metadatas"][0][i]),
            "distance": results["distances"][0][i],
            "rank": i + 1
        }
        formatted_results.append(result)
    
    return formatted_results
```

### Batch Operations for Efficiency
**Goal**: Handle large numbers of chunks efficiently during indexing.

```python
def add_chunks_batch(self, chunks: List[Dict], batch_size: int = 100) -> None:
    """Add chunks in batches for memory efficiency"""
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare batch data
        ids = [chunk["id"] for chunk in batch]
        documents = [chunk["document"] for chunk in batch]
        metadatas = [
            self.prepare_metadata_for_chromadb(chunk["metadata"]) 
            for chunk in batch
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        # Log progress
        logger.info(f"Added batch {i//batch_size + 1}, chunks {i+1}-{min(i+batch_size, len(chunks))}")
```

## Configuration (.env Variables)

```bash
# ChromaDB Configuration
CHROMADB_PERSIST_DIR=./chromadb_data
CHROMA_COLLECTION_NAME=api_knowledge
CHROMA_RESET_ON_START=false        # Whether to clear collection on startup

# Embedding Model Configuration
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=mps              # mps, cuda, or cpu
EMBEDDING_CACHE_DIR=./model_cache

# Search Configuration
VECTOR_SEARCH_LIMIT=5             # Default search result limit
SIMILARITY_THRESHOLD=0.3          # Minimum similarity for results
BATCH_SIZE=100                    # Batch size for bulk operations

# Performance Tuning
CHROMADB_THREADS=4               # Number of threads for ChromaDB
EMBEDDING_BATCH_SIZE=32          # Batch size for embedding generation
```

## Definition of Done

### Functional Requirements
1. **Collection Management**: Create and configure ChromaDB collection with proper settings
2. **Embedding Integration**: Successfully load and use Qwen3-0.6B with MPS acceleration
3. **Chunk Storage**: Store all sample OpenAPI chunks with complete metadata
4. **Search Functionality**: Perform semantic search with configurable filters and limits
5. **ID Retrieval**: Fast lookup of chunks by exact ID for reference resolution
6. **Data Persistence**: Persist collection data and restore on restart

### Measurable Success Criteria
1. **Storage Capacity**: Successfully store all chunks from sample specs (~245 total chunks)
2. **Search Performance**: <150ms average query time (including embedding generation)
3. **Memory Efficiency**: Handle embedding generation with <4GB RAM usage
4. **Accuracy**: Semantic search returns relevant results for test queries
5. **ID Lookup Speed**: <5ms average time for ID-based chunk retrieval
6. **Metadata Integrity**: 100% metadata round-trip accuracy (store → retrieve → verify)

### Integration Test Scenarios
1. **Full Indexing**: Store all chunks from OpenAPI Processor output, verify no data loss
2. **Search Quality**: Test semantic search with known queries, verify relevant results in top-5
3. **Filter Testing**: Search with metadata filters (by source_file, type), verify correct filtering
4. **ID Resolution**: Retrieve chunks by ID for reference expansion, verify completeness
5. **Persistence**: Restart component, verify collection data survives and loads correctly
6. **Performance**: Measure query times under load, verify within targets

## Implementation Guidelines

### Code Structure
```python
# Suggested file organization
vector_store/
├── __init__.py
├── chromadb_manager.py      # ChromaDB collection management
├── embedding_function.py    # Custom Qwen3 embedding function
├── search_engine.py         # Search operations and filtering
├── metadata_handler.py      # Metadata conversion and restoration
└── vector_store.py         # Main orchestration class
```

### Key Classes
```python
class VectorStoreManager:
    def __init__(self, config: Config):
        """Initialize ChromaDB client and embedding function"""
        
    def setup_collection(self) -> None:
        """Create or connect to ChromaDB collection"""
        
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Store chunks with metadata and generate embeddings"""
        
    def search(self, query: str, **kwargs) -> List[Dict]:
        """Semantic search with filtering"""
        
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """Direct retrieval by chunk IDs"""

class EmbeddingManager:
    def __init__(self, model_name: str, device: str):
        """Initialize embedding model with proper device acceleration"""
        
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Encode documents for storage"""
        
    def encode_query(self, query: str) -> List[float]:
        """Encode query with proper prompt for search"""
```

### Error Handling
- **Model Loading**: Graceful fallback if MPS/CUDA unavailable (use CPU)
- **Collection Conflicts**: Handle existing collections (reset or append based on config)
- **Memory Management**: Monitor memory usage during large batch operations
- **Embedding Failures**: Log and skip problematic documents, continue processing
- **Search Errors**: Return empty results for invalid queries, log issues

### Performance Optimizations
- **Lazy Loading**: Load embedding model only when needed
- **Batch Processing**: Use optimal batch sizes for embedding generation
- **Memory Monitoring**: Track and limit memory usage during operations
- **Connection Pooling**: Reuse ChromaDB connections efficiently
- **Caching**: Cache frequently accessed embeddings and search results

## Integration Points

### Upstream Dependencies
- **OpenAPI Processor**: Receives chunks for storage and indexing
- **Configuration**: Uses .env settings for all behavior control

### Downstream Dependencies
- **Knowledge Retriever**: Provides search and lookup services
- **Evaluation Framework**: Supports testing with stored chunks

### Data Contract Validation
```python
def validate_chunk_storage(chunk: Dict) -> bool:
    """Validate chunk can be stored in ChromaDB"""
    # Check required fields
    # Validate ID format
    # Verify metadata compatibility
    # Test document text quality

def validate_search_results(results: List[Dict]) -> bool:
    """Validate search results format and completeness"""
    # Check result structure
    # Verify metadata restoration
    # Validate similarity scores
    # Test ranking order
```

## Testing Requirements

### Unit Tests
- Test embedding function with various document types
- Test metadata conversion (complex → ChromaDB → restored)
- Test search filtering with different criteria
- Test batch operations with various sizes

### Integration Tests
- Store and retrieve complete sample dataset
- Semantic search with known good/bad results
- Performance testing with large datasets
- Persistence testing across restarts

### Performance Tests
- Measure embedding generation speed
- Test search performance under concurrent load
- Memory usage during batch indexing
- Query response time distribution

### Quality Assurance
- Verify embedding quality with semantic similarity tests
- Test search relevance with manual evaluation
- Validate metadata integrity across operations
- Check collection consistency after restart

This specification provides complete guidance for implementing the Vector Store Manager using the validated ChromaDB integration approach and embedding model configuration from the prototype testing.