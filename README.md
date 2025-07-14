# Knowledge Server

A specialized MCP (Model Context Protocol) server that makes large technical documentation accessible to LLMs through intelligent chunking and retrieval. Originally designed for OpenAPI specifications but now supports general knowledge management including markdown documents.

## Overview

The Knowledge Server solves the fundamental problem of large technical documentation (OpenAPI specs, markdown docs, etc.) that cannot fit in LLM context windows. It provides intelligent chunking, semantic search, and automatic reference resolution to deliver complete, accurate information through a simple interface.

### Key Features

- **Universal Document Support**: OpenAPI specifications (JSON/YAML) and markdown documents
- **Intelligent Chunking**: Context-aware splitting with reference tracking
- **Semantic Search**: Vector-based similarity search with configurable embedding models
- **Reference Resolution**: Automatic expansion of related content and dependencies
- **Research Agent**: Intelligent ReAct agent for comprehensive analysis
- **MCP Integration**: Standard Model Context Protocol server for LLM tools
- **Configurable LLM Support**: Local models (GGUF) and cloud providers (AWS Bedrock)

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd knowledge-server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required: Document directories
OPENAPI_SPECS_DIR=/path/to/your/openapi/specs

# Vector Store Configuration
VECTOR_STORE_DIR=./data/vectorstore
VECTOR_STORE_COLLECTION=knowledge_base
EMBEDDING_MODEL=dunzhang/stella_en_1.5B_v5
EMBEDDING_DEVICE=mps  # mps, cpu, cuda
MAX_TOKENS=8192

# API Index Configuration  
API_INDEX_PATH=./data/api_index.json

# Knowledge Retriever Configuration
RETRIEVAL_MAX_PRIMARY_RESULTS=5
RETRIEVAL_MAX_TOTAL_CHUNKS=15
RETRIEVAL_MAX_DEPTH=3
RETRIEVAL_TIMEOUT_MS=5000
CONTEXT_PRIORITIZE_PRIMARY=true

# MCP Server Configuration
MCP_SERVER_NAME=Knowledge Server
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000

# Processing Configuration
SKIP_HIDDEN_FILES=true
SUPPORTED_EXTENSIONS=.json,.yaml,.yml
LOG_PROCESSING_PROGRESS=true
```

### Index Your Documents

```bash
# Index both OpenAPI specs and markdown documents
knowledge-server index

# Index only OpenAPI specifications
knowledge-server index --skip-markdown

# Index only markdown documents  
knowledge-server index --skip-openapi

# Specify custom markdown directory
knowledge-server index --markdown-dir /path/to/docs

# Control chunk size for markdown (default: 1000 tokens, max: 8000)
knowledge-server index --max-tokens 1500
```

### Start MCP Server

```bash
# Start the MCP server
knowledge-server serve

# With verbose output for debugging
knowledge-server serve -v
```

## Usage

### As MCP Server

The primary use case is as an MCP server providing two main tools:

#### `search_api(query, max_response_length, max_chunks, include_references, max_depth)`
Search your indexed knowledge base and return relevant chunks.

#### `research_api(question)` 
Use the intelligent ReAct agent for comprehensive analysis and implementation guidance.

### Direct CLI Usage

For testing and development:

```bash
# Ask questions about your documentation
knowledge-server ask "How do I authenticate with the user API?"

# Use the research agent for comprehensive analysis
knowledge-server research "What are the best practices for pagination in this API?"

# Advanced search options
knowledge-server ask "API rate limits" --max-chunks 30 --max-depth 2 --no-references
```

## MCP Client Configuration

**Important**: All MCP configurations must use the Python executable from the virtual environment (`venv/bin/python`) to ensure all dependencies are available.

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": [
        "-m", "src.mcp_server.server"
      ],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "PYTHONPATH": "/path/to/knowledge-server"
      }
    }
  }
}
```

Or using the convenience script (make sure it's executable: `chmod +x run_server.sh`):

```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/run_server.sh",
      "cwd": "/path/to/knowledge-server"
    }
  }
}
```

### VS Code with Cline

Add to your Cline MCP settings:

```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "PYTHONPATH": "/path/to/knowledge-server"
      }
    }
  }
}
```

### Generic MCP Client

The server implements the standard MCP protocol and can be used with any compatible client:

```bash
# Direct execution
cd /path/to/knowledge-server
source venv/bin/activate
python -m src.mcp_server.server
```

## Document Processing

### OpenAPI Specifications

The server processes OpenAPI 3.0/3.1 specifications:

- **Supported formats**: JSON and YAML
- **Intelligent chunking**: Operations grouped with related schemas
- **Reference resolution**: Automatic $ref expansion
- **Metadata extraction**: Comprehensive tagging and categorization

### Markdown Documents

Supports structured markdown processing:

- **Header-based chunking**: Sections split at configurable token limits
- **Reference tracking**: Cross-document links and references
- **Navigation building**: Automatic section hierarchy
- **Content analysis**: Semantic categorization

## Architecture

The Knowledge Server uses a modular architecture:

1. **Document Processors**: Handle OpenAPI and markdown parsing
2. **Vector Store Manager**: ChromaDB integration with configurable embeddings
3. **Knowledge Retriever**: Two-stage retrieval with reference expansion
4. **Research Agent**: LangGraph-based intelligent analysis
5. **MCP Server**: Standard protocol interface for LLM tools

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation.

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAPI_SPECS_DIR` | Directory containing OpenAPI specs | Required |
| `VECTOR_STORE_DIR` | Vector store persistence directory | `./data/vectorstore` |
| `VECTOR_STORE_COLLECTION` | Vector store collection name | `knowledge_base` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `dunzhang/stella_en_1.5B_v5` |
| `EMBEDDING_DEVICE` | Device for embeddings | `mps` |
| `MAX_TOKENS` | Maximum tokens per chunk | `8192` |
| `API_INDEX_PATH` | Path to API index file | `./data/api_index.json` |
| `RETRIEVAL_MAX_PRIMARY_RESULTS` | Max primary search results | `5` |
| `RETRIEVAL_MAX_TOTAL_CHUNKS` | Max total chunks retrieved | `15` |
| `RETRIEVAL_MAX_DEPTH` | Max reference expansion depth | `3` |
| `RETRIEVAL_TIMEOUT_MS` | Retrieval timeout in milliseconds | `5000` |
| `MCP_SERVER_NAME` | MCP server display name | `Knowledge Server` |
| `MCP_SERVER_HOST` | MCP server host | `localhost` |
| `MCP_SERVER_PORT` | MCP server port | `8000` |

### Embedding Models

Supported embedding models (via sentence-transformers):

- `dunzhang/stella_en_1.5B_v5` (default) - High-quality English embeddings
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight
- `sentence-transformers/all-mpnet-base-v2` - Good balance of speed and quality
- `BAAI/bge-large-en-v1.5` - State-of-the-art English embeddings

### Device Support

- **cpu**: CPU-only processing (default, most compatible)
- **mps**: Apple Silicon GPU acceleration (recommended for M1/M2/M3 Macs)
- **cuda**: NVIDIA GPU acceleration
- **auto**: Automatically select best available device

## Development

### Testing

```bash
# Run all tests
source venv/bin/activate
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/openapi_processor/ -v
python -m pytest tests/markdown_processor/ -v
python -m pytest tests/integration/ -v
```

### Code Quality

```bash
# Format and lint
source venv/bin/activate
black src/ tests/
isort src/ tests/
python -m pytest tests/
```

### Project Structure

```
knowledge-server/
├── src/
│   ├── cli/                    # Command-line interface
│   ├── mcp_server/            # MCP protocol server
│   ├── openapi_processor/     # OpenAPI document processing
│   ├── markdown_processor/    # Markdown document processing
│   ├── vector_store/          # ChromaDB integration
│   ├── retriever/            # Knowledge retrieval engine
│   ├── research_agent/       # Intelligent analysis agent
│   ├── llm/                  # LLM provider abstraction
│   └── utils/                # Shared utilities
├── tests/                    # Comprehensive test suite
├── data/                     # Generated indices and vector store
├── docs/                     # Documentation and specifications
└── samples/                  # Example documents and configurations
```

## Troubleshooting

### Common Issues

**MCP server not starting**
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify `.env` file configuration
- Ensure directories in config exist and are readable

**No search results**
- Run indexing first: `knowledge-server index`
- Check that document directories contain supported files
- Verify ChromaDB persistence directory is writable

**Embedding model download fails**
- Check internet connectivity
- Try a different embedding model
- Use CPU device if GPU memory is insufficient

**LLM provider errors**
- Verify AWS credentials and permissions for Bedrock
- Check local model path and GGUF format for local provider
- Ensure sufficient memory for local models

### Performance Optimization

- Use GPU acceleration for embeddings when available
- Adjust `VECTOR_SEARCH_LIMIT` based on your use case
- Consider using smaller embedding models for faster indexing
- Increase `max_chunks` parameter for comprehensive but slower searches

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for development guidelines and coding standards.