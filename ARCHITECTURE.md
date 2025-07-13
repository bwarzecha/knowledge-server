# Knowledge Server Architecture

## Overview

The Knowledge Server is a specialized MCP (Model Context Protocol) server that makes large OpenAPI specifications accessible to LLMs through intelligent chunking and retrieval. It solves the fundamental problem of 200+ enterprise API specs (50KB-2MB each) that cannot fit in LLM context windows, providing a simple `askAPI("your question")` interface that returns complete, accurate API information.

The system transforms complex OpenAPI specifications into a searchable knowledge base using semantic embeddings and automatic reference resolution, ensuring that when an LLM asks about an API endpoint, it receives not just the endpoint definition but all related schemas, error codes, and dependencies needed for accurate code generation.

## System Tenets

- **Keep it Simple**: No over-engineering. Prefer straightforward solutions over complex abstractions.
- **Independent & Testable**: Components have clear contracts and can be tested in isolation.
- **Integration Testing**: Avoid mocks. Use real data and well-defined interfaces.
- **DRY Principle**: Eliminate repetition. Design reusable functions within components.
- **Short & Beautiful**: Target <200 lines per file. Prefer few clear classes over many complex ones.
- **Meaningful Comments**: Comment when it adds value, avoid excessive documentation.

## Component Architecture

### 1. OpenAPI Processor ✅ **IMPLEMENTED**
**Responsibility**: Transforms OpenAPI specifications into searchable chunks with metadata and reference tracking.

This component is fully implemented with a comprehensive 5-phase pipeline:
- **Scanner**: Discovers OpenAPI files in directories (JSON/YAML)
- **Parser**: Loads and validates file formats
- **Validator**: Ensures OpenAPI specification compliance
- **Extractor**: Extracts operations, schemas, and metadata
- **Graph Builder**: Builds dependency graphs for reference resolution
- **Chunk Assembler**: Creates final chunks with inlined content
- **Chunk Splitter**: Handles large chunks intelligently

The processor implements the file-path-based ID strategy (`{filename}:{natural_name}`) and creates chunks with `ref_ids` for dependency tracking.

**Interaction**: Reads from specs directory, outputs chunks to Vector Store Manager. Generates dependency graphs used by Knowledge Retriever.

### 2. Vector Store Manager ✅ **IMPLEMENTED**
**Responsibility**: Manages ChromaDB operations including document storage, embedding generation, and filtered semantic search.

Fully implemented with modular design:
- **ChromaDB Utils**: Direct ChromaDB operations (collections, search, storage)
- **Embedding Utils**: Model loading and text encoding (supports sentence-transformers)
- **Metadata Utils**: Serialization/deserialization for ChromaDB storage
- **Vector Store Manager**: Main orchestrator with environment-based configuration

Supports configurable embedding models (default: Qwen/Qwen3-Embedding-0.6B) with MPS/CUDA acceleration, filtered search, and efficient ID-based lookups.

**Interaction**: Receives chunks from OpenAPI Processor for indexing. Serves search requests from Knowledge Retriever with semantic similarity and metadata filtering.

### 3. Knowledge Retriever ✅ **IMPLEMENTED**  
**Responsibility**: Orchestrates intelligent retrieval by combining semantic search with automatic reference expansion.

Core intelligence implemented with three sub-components:
- **Knowledge Retriever**: Main orchestrator implementing two-stage retrieval
- **Reference Expander**: Follows `ref_ids` to resolve dependencies with circular reference protection
- **Context Assembler**: Assembles final context packages with statistics
- **Data Classes**: Configuration and result structures

Implements depth limiting, count limiting, and comprehensive context assembly for complete query responses.

**Interaction**: Uses Vector Store Manager for search and lookups. Returns complete context packages with primary results and resolved dependencies.

### 4. LLM Integration ✅ **IMPLEMENTED**
**Responsibility**: Provides pluggable LLM integration for various providers.

Currently implemented providers:
- **Abstract Provider Interface**: Common interface for all LLM providers
- **Bedrock Provider**: AWS Bedrock integration with boto3
- **Local Provider**: Local model support (GGUF, transformers)
- **LLM Client**: Main interface for LLM operations

**Interaction**: Used by Query Expansion and future MCP Server for natural language processing.

### 5. Query Expansion ✅ **IMPLEMENTED**
**Responsibility**: Enhances search queries using API context and LLM processing.

Implemented components:
- **Query Expander**: Uses pre-built API index with LLM for query enhancement
- **Index Builder**: Builds compact API indexes for context-aware expansion

**Interaction**: Uses LLM providers to expand queries before semantic search.

### 6. MCP Server ❌ **NOT YET IMPLEMENTED**
**Responsibility**: Will expose the `askAPI()` interface and handle LLM integration for question answering.

**Status**: Planned implementation
- Will provide clean external interface for LLMs and developers
- Will integrate Knowledge Retriever with LLM providers
- Will handle prompt engineering and response formatting
- Will support streaming and batch response modes

### 7. Evaluation Framework ❌ **NOT YET IMPLEMENTED** 
**Responsibility**: Will provide automated testing and quality measurement for the knowledge retrieval system.

**Status**: Planned implementation  
- Will implement evaluation methodology using sample OpenAPI specs
- Will measure retrieval completeness, reference resolution accuracy
- Will provide regression detection and quality metrics

## Data Flow

### Current Implementation (Core Pipeline)
```
OpenAPI Specs Directory 
    ↓ (parse & chunk)
OpenAPI Processor 
    ↓ (chunks with metadata)
Vector Store Manager 
    ↓ (store & index)
ChromaDB with Embeddings
    ↓ (search requests)
Knowledge Retriever 
    ↓ (expanded context)
Knowledge Context Response
```

### With Query Expansion (Optional)
```
User Query
    ↓ (expand with API context)
Query Expander + LLM
    ↓ (enhanced query)
Knowledge Retriever
    ↓ (search + expand)
Complete Knowledge Context
```

### Future Complete Flow (with MCP Server)
```
askAPI(question)
    ↓ (optional query expansion)
Knowledge Retriever 
    ↓ (expanded context)
MCP Server + LLM
    ↓ (formatted response)
Final Answer
```

**Current Detailed Flow**:
1. **Indexing Phase**: OpenAPI Processor scans specs directory, creates chunks with file-path IDs and `ref_ids`, sends to Vector Store Manager for embedding and storage
2. **Query Phase**: Applications can directly call Knowledge Retriever with search queries
3. **Retrieval Phase**: Knowledge Retriever performs semantic search via Vector Store Manager, expands results by following `ref_ids`, assembles complete context with statistics
4. **Optional Enhancement**: Query Expander can enhance queries using API context and LLM before retrieval

## Technology Stack

- **Python 3.9+**: Modern Python with type hints and comprehensive testing
- **ChromaDB**: Vector database for embedding storage and similarity search  
- **Sentence Transformers**: Configurable embedding models (default: Qwen/Qwen3-Embedding-0.6B)
- **Device Support**: MPS, CUDA, and CPU acceleration for embeddings
- **LLM Integration**: Pluggable support for local models (GGUF) and AWS Bedrock (boto3)
- **Testing**: pytest with 130+ tests covering integration and component testing
- **Code Quality**: black, isort, flake8 for formatting and linting
- **Dependencies**: sentence-transformers, pydantic, python-dotenv, PyYAML, tiktoken, chromadb

## Development Workflow

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Configuration
Create `.env` file with:
```
# Required
OPENAPI_SPECS_DIR=/path/to/specs
CHROMADB_PERSIST_DIR=./chromadb_data

# Embedding Model
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=mps  # or cuda, cpu

# LLM Configuration  
LLM_PROVIDER=local  # or aws_bedrock
LOCAL_MODEL_PATH=/path/to/model  # if using local LLM
AWS_REGION=us-east-1  # if using Bedrock
AWS_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0  # if using Bedrock

# ChromaDB Settings
CHROMA_COLLECTION_NAME=api_knowledge
VECTOR_SEARCH_LIMIT=5
```

### Testing Strategy (Current Implementation)
- **Component Tests**: 130+ tests covering all implemented components with real data
- **Integration Tests**: End-to-end workflows using sample OpenAPI specs
- **Vector Store Tests**: Real embedding models and ChromaDB operations  
- **Reference Resolution Tests**: Complex dependency graphs and circular reference handling
- **No Mocks**: Following project tenets, tests use real components and data
- **Performance Validation**: Retrieval speed and token estimation accuracy testing

**Test Coverage**:
- OpenAPI Processor: 40+ tests (parsing, validation, chunking, reference resolution)
- Vector Store: 25+ tests (embedding, storage, search operations)
- Knowledge Retriever: 15+ tests (search, expansion, context assembly)
- Integration: 6+ tests (end-to-end workflows)

### Code Organization
```
knowledge-server/
├── ARCHITECTURE.md                    # This document
├── CLAUDE.md                         # Development guidelines
├── components/                        # Component specifications
├── src/
│   ├── openapi_processor/            # ✅ OpenAPI parsing and chunking (11 modules)
│   ├── vector_store/                 # ✅ ChromaDB operations (4 modules)
│   ├── retriever/                    # ✅ Search and reference expansion (4 modules)
│   ├── llm/                          # ✅ LLM provider abstraction (3 modules)
│   └── query_expansion/              # ✅ Query enhancement (2 modules)
├── data/                             # API indexes and cached data
├── docs/                            # Documentation and design archives
├── models/                          # Local model storage
├── open-api-small-samples/          # Test OpenAPI specifications
├── prototypes/                      # Research and validation code
├── samples/                         # Additional test data
├── tests/                           # ✅ Comprehensive test suites (130+ tests)
├── pyproject.toml                   # Project configuration and dependencies
└── requirements.txt                 # Core dependencies
```

**Note**: MCP Server and Evaluation Framework are planned but not yet implemented.

## Key Design Decisions

### File-Path-Based IDs
**Decision**: Use `{filename}:{natural_name}` format for chunk IDs
**Rationale**: Eliminates collision concerns across API versions, services, and schemas while maintaining human readability
**Example**: `amazon-ads-sponsored-display-v3:listCampaigns`

### Hybrid Chunking Strategy  
**Decision**: Endpoint chunks with inlined simple schemas + separate complex schema chunks
**Rationale**: Balances retrieval efficiency (fewer chunks to fetch) with completeness (all dependencies available)
**Validation**: Achieved 80% query completeness in prototype testing

### Semantic Search + Reference Expansion
**Decision**: Two-stage retrieval (search then expand) rather than pre-flattening or query-time joining
**Rationale**: Maintains clean separation between content search and dependency resolution, enables efficient caching and filtering
**Performance**: Validated with 100ms average query time in ChromaDB integration testing

### Configuration-Driven LLM Integration
**Decision**: Pluggable LLM interface supporting both local models and cloud APIs
**Rationale**: Enables cost-effective development (local) and production deployment (cloud) without code changes
**Implementation**: Abstract LLM interface with provider-specific implementations

### .env-Based Configuration
**Decision**: Centralized configuration in environment variables
**Rationale**: Simplifies deployment, enables easy testing with different settings, follows 12-factor app principles
**Scope**: All component behavior configurable without code changes

## Success Metrics

### Current Implementation Status
- **Reference Resolution**: ✅ 100% success rate - no broken ref_ids in test suite
- **Test Coverage**: ✅ 130+ tests passing with real data and components
- **Component Integration**: ✅ All core components working together seamlessly
- **Performance**: ✅ Efficient retrieval with depth/count limiting and circular reference protection
- **Embedding Support**: ✅ Multiple models supported with MPS/CUDA acceleration

### Target Metrics (for complete system)
- **Retrieval Completeness**: >90% of test queries return all necessary schemas and dependencies
- **Response Accuracy**: Generated examples pass schema validation  
- **Performance**: <200ms average query response time including LLM processing
- **Scalability**: Handle 200+ API specifications without degradation

## Implementation Status & Roadmap

### ✅ Completed Core Components (Production Ready)
- **OpenAPI Processing Pipeline**: Complete 5-phase processing with all subcomponents
- **Vector Store Operations**: Full ChromaDB integration with configurable embedding models
- **Knowledge Retrieval**: Two-stage retrieval with reference expansion and context assembly
- **LLM Integration**: Pluggable providers (Bedrock, Local models)
- **Query Enhancement**: LLM-powered query expansion with API context

### 🚧 Next Priority (Immediate Implementation)
- **MCP Server**: Expose `askAPI()` interface for LLM integration
- **Evaluation Framework**: Automated quality measurement and regression testing

### 🔮 Future Enhancements
- **Advanced Evaluation**: LLM judges for automated quality assessment
- **Cross-API Search**: Enhanced metadata and filtering for multi-API queries  
- **Performance Optimization**: Caching layers and embedding model upgrades
- **Extended LLM Support**: Additional provider integrations

## Conclusion

The Knowledge Server has a solid, production-ready foundation with 5 of 7 planned components fully implemented and thoroughly tested. The modular architecture enables independent development of remaining components while maintaining the core tenets of simplicity, testability, and reliability.

**Current State**: A functional knowledge retrieval system that can process OpenAPI specs, store them in vector databases, and provide intelligent context-aware search with automatic reference resolution.

**Next Steps**: Implement MCP Server to provide the clean `askAPI()` interface that makes this powerful backend accessible to LLMs and developers.