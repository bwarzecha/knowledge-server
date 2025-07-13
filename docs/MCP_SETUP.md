# MCP Server Setup Guide

This guide covers how to configure the Knowledge Server as an MCP (Model Context Protocol) server with popular LLM tools and clients.

## Quick Setup

The Knowledge Server implements the standard MCP protocol and provides two main tools:
- `search_api`: Search your indexed knowledge base 
- `research_api`: Use intelligent ReAct agent for comprehensive analysis

## Supported Clients

### Claude Desktop

Claude Desktop is the official desktop application for Claude with built-in MCP support.

#### Configuration

**Location of config file:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Method 1: Using Python directly**
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

**Method 2: Using the convenience script**
```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/run_server.sh"
    }
  }
}
```

**Method 3: Using installed package (if installed via pip)**
```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "knowledge-server",
      "args": ["serve"],
      "cwd": "/path/to/knowledge-server"
    }
  }
}
```

#### Usage in Claude Desktop

After configuration, restart Claude Desktop. You can then use the tools:

```
@knowledge-server search_api "How do I authenticate with the API?"
```

```
@knowledge-server research_api "What are the best practices for error handling in this API?"
```

### VS Code with Cline Extension

Cline is a popular VS Code extension that supports MCP servers.

#### Configuration

1. Install the Cline extension in VS Code
2. Open VS Code settings (Cmd/Ctrl + ,)
3. Search for "Cline MCP"
4. Add the Knowledge Server configuration:

```json
{
  "cline.mcpServers": {
    "knowledge-server": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "PYTHONPATH": "/path/to/knowledge-server"
      }
    }
  }
}
```

#### Alternative: settings.json

Add to your VS Code `settings.json`:

```json
{
  "cline.mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server"
    }
  }
}
```

### Continue.dev

Continue.dev is another popular coding assistant that supports MCP.

#### Configuration

Add to your Continue configuration file (`.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "knowledge-server",
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server"
    }
  ]
}
```

### Cursor IDE

Cursor IDE with MCP support configuration:

```json
{
  "mcp": {
    "servers": {
      "knowledge-server": {
        "command": "/path/to/knowledge-server/venv/bin/python",
        "args": ["-m", "src.mcp_server.server"],
        "cwd": "/path/to/knowledge-server"
      }
    }
  }
}
```

### Generic MCP Client

For any MCP-compatible client, use these connection details:

**Server Command**: `/path/to/knowledge-server/venv/bin/python`
**Arguments**: `["-m", "src.mcp_server.server"]`
**Working Directory**: `/path/to/knowledge-server`

## Tool Reference

### search_api

Search your indexed knowledge base and return relevant chunks.

**Parameters:**
- `query` (string, required): Natural language search query
- `max_response_length` (int, default: 4000): Maximum response length in tokens
- `max_chunks` (int, default: 50): Maximum number of chunks to retrieve
- `include_references` (bool, default: true): Whether to follow references between chunks
- `max_depth` (int, default: 3): Maximum depth for reference expansion (1-5)

**Example:**
```
search_api("How do I authenticate with the user management API?", max_chunks=30, max_depth=2)
```

### research_api

Use the intelligent ReAct agent for comprehensive analysis and implementation guidance.

**Parameters:**
- `question` (string, required): Research question about your documentation

**Example:**
```
research_api("What are the best practices for implementing pagination across all available APIs?")
```

## Environment Setup

### Prerequisites

Before configuring MCP clients, ensure:

1. **Knowledge Server is installed**:
   ```bash
   cd /path/to/knowledge-server
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration file exists** (`.env`):
   ```env
   OPENAPI_SPECS_DIR=/path/to/your/specs
   CHROMADB_PERSIST_DIR=./data/vectorstore
   EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
   LLM_PROVIDER=aws_bedrock
   AWS_REGION=us-east-1
   AWS_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   ```

3. **Documents are indexed**:
   ```bash
   knowledge-server index
   ```

4. **Server can start**:
   ```bash
   knowledge-server serve -v
   ```

### Path Configuration

Replace `/path/to/knowledge-server` in all examples with your actual installation path:

**Find your path:**
```bash
cd knowledge-server
pwd
```

**Common locations:**
- Development: `/Users/username/dev/knowledge-server`
- System-wide: `/opt/knowledge-server`
- User-local: `~/tools/knowledge-server`

## Troubleshooting

### Server Won't Start

**Check dependencies:**
```bash
cd /path/to/knowledge-server
source venv/bin/activate
pip install -r requirements.txt
```

**Verify configuration:**
```bash
knowledge-server serve -v
```

**Check logs:** Look for error messages during startup.

### No Tools Available in Client

1. **Restart the client** after adding MCP configuration
2. **Check the server path** is correct and executable
3. **Verify working directory** contains the project files
4. **Test manually**:
   ```bash
   cd /path/to/knowledge-server
   source venv/bin/activate
   python -m src.mcp_server.server
   ```

### Tools Return Empty Results

1. **Index your documents first**:
   ```bash
   knowledge-server index
   ```

2. **Check document directories exist** and contain files
3. **Verify vector store was created**: Look for files in `CHROMADB_PERSIST_DIR`

### Permission Errors

**Make scripts executable:**
```bash
chmod +x /path/to/knowledge-server/run_server.sh
```

**Check directory permissions:**
```bash
ls -la /path/to/knowledge-server
```

### Performance Issues

1. **Use GPU acceleration** if available:
   ```env
   EMBEDDING_DEVICE=cuda  # or mps for Apple Silicon
   ```

2. **Adjust search limits**:
   ```env
   VECTOR_SEARCH_LIMIT=20  # Reduce for faster responses
   ```

3. **Use smaller embedding model**:
   ```env
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

## Advanced Configuration

### Custom Environment Variables

Pass environment variables through MCP client configuration:

```json
{
  "mcpServers": {
    "knowledge-server": {
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
        "VECTOR_SEARCH_LIMIT": "30",
        "EMBEDDING_DEVICE": "cuda"
      }
    }
  }
}
```

### Multiple Instances

Run multiple Knowledge Server instances for different document sets:

```json
{
  "mcpServers": {
    "api-docs": {
      "command": "/path/to/knowledge-server/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "OPENAPI_SPECS_DIR": "/path/to/api/docs",
        "CHROMA_COLLECTION_NAME": "api_docs"
      }
    },
    "user-guides": {
      "command": "/path/to/knowledge-server/venv/bin/python", 
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/knowledge-server",
      "env": {
        "MARKDOWN_DOCS_DIR": "/path/to/user/guides",
        "CHROMA_COLLECTION_NAME": "user_guides"
      }
    }
  }
}
```

### Security Considerations

1. **File Permissions**: Ensure only authorized users can access the knowledge server directory
2. **Network Access**: The MCP server runs locally and doesn't expose network ports
3. **Environment Variables**: Avoid putting sensitive data in MCP client configuration files
4. **Document Access**: The server can access any files in configured directories

## Getting Help

- **Test the server standalone**: `knowledge-server serve -v`
- **Check client logs**: Most MCP clients show connection status and errors
- **Validate configuration**: Use `knowledge-server ask "test query"` to verify functionality
- **Review documentation**: See the main [README.md](../README.md) for detailed setup instructions