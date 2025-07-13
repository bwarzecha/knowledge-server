"""FastMCP server for knowledge server API documentation queries."""

from mcp.server.fastmcp import FastMCP

from src.cli.config import Config
from src.utils.logging_config import setup_logging

from .askAPI import askAPI
from .shared_resources import initialize_shared_resources

# Setup logging for MCP (silent mode - ERROR level only)
setup_logging(verbose=False)

# Create MCP server
mcp = FastMCP("Knowledge Server")


# Register the askAPI tool
@mcp.tool()
async def ask_api(
    query: str,
    max_response_length: int = 4000,
    max_chunks: int = 50,
    include_references: bool = True,
    max_depth: int = 3,
) -> str:
    """
    Ask questions about API documentation and get comprehensive answers.

    Args:
        query: Natural language question about API usage, endpoints, schemas, or examples
        max_response_length: Maximum response length in tokens (for LLM context window)
        max_chunks: Maximum number of chunks to retrieve from knowledge base
        include_references: Whether to follow references between chunks for complete context
        max_depth: Maximum depth for reference expansion (1-5, default 3)

    Returns:
        Formatted response with relevant API documentation, trimmed to fit within token limit
    """
    return await askAPI(
        query=query,
        max_response_length=max_response_length,
        max_chunks=max_chunks,
        include_references=include_references,
        max_depth=max_depth,
    )


def start_server(config: Config):
    """Start MCP server with pre-built indices."""
    # Initialize shared resources silently (no stdout prints for MCP)
    initialize_shared_resources(config)

    # Start the MCP server
    mcp.run()


if __name__ == "__main__":
    # Load config for standalone execution
    config = Config()
    start_server(config)
