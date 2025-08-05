"""FastMCP server for knowledge server API documentation queries."""

from mcp.server.fastmcp import FastMCP

from src.cli.config import Config
from src.utils.logging_config import setup_logging

from ..research_agent import research_api_question
from .searchAPI import searchAPI
from .shared_resources import initialize_shared_resources

# Setup logging for MCP (silent mode - ERROR level only)
setup_logging(verbose=False)

# Create MCP server
mcp = FastMCP("Knowledge Server")


# Register the searchAPI tool
@mcp.tool()
async def search_api(
    query: str,
    max_response_length: int = 4000,
    max_chunks: int = 50,
    include_references: bool = True,
    max_depth: int = 3,
    exclude_chunks: str = "",
) -> str:
    """
    WHEN TO USE: Direct lookups for specific schema fields, endpoint details, or known API elements.
    Use when you know exactly what you're looking for in a specific schema or API endpoint.

    BEST FOR:
    - Specific field definitions in known schemas
    - Direct API endpoint parameters and responses
    - Exact error codes and status messages
    - Authentication/authorization details for specific endpoints
    - Simple field value constraints or enums

    EXAMPLES:
    - "What fields are required in the Campaign schema?"
    - "Show me the bidding parameters for Sponsored Products"
    - "What authentication headers does the POST /campaigns endpoint require?"
    - "What are the possible values for campaign state field?"

    NOT FOR COMPLEX ANALYSIS - Use research_api instead for:
    - Understanding relationships between multiple schemas
    - "What do I need to create a Sponsored Display campaign?" (needs campaign + ad group + targeting + ad schemas)
    - "What creative types exist in Sponsored Display?" (requires traversing multiple schema references)

    EXCLUDE_CHUNKS PARAMETER USAGE:
    Use exclude_chunks to avoid getting the same documentation chunks in follow-up questions within the same conversation.

    WHEN TO USE exclude_chunks:
    - After receiving results from a previous search_api call in the same conversation
    - When asking related but different questions about the same topic
    - To get fresh, non-duplicate content for comprehensive coverage

    HOW TO USE exclude_chunks:
    1. First call: search_api("authentication methods") → Returns chunks: "auth_basic_001", "auth_oauth_002", "auth_token_003"
    2. Follow-up call: search_api("authentication examples", exclude_chunks="auth_basic_001,auth_oauth_002,auth_token_003")
    3. Next call: search_api("auth error handling", exclude_chunks="auth_basic_001,auth_oauth_002,auth_token_003,auth_examples_004")

    EXTRACT CHUNK IDs FROM RESPONSES:
    Look for chunk IDs in the response footer like: "Sources: 3 chunks" or in detailed responses.
    Each chunk has an ID format like: schema_name_001, endpoint_post_campaigns_002, etc.

    Args:
        query: Specific search query (use exact schema names, field names, endpoint paths when known)
        max_response_length: Token limit for response (default 4000 works for most direct lookups)
        max_chunks: Number of documentation chunks to search (default 50 sufficient for direct queries)
        include_references: Set True for related context, False for just direct matches
        max_depth: Reference following depth (1=direct only, 3=default, 5=comprehensive context)
        exclude_chunks: Comma-separated chunk IDs to exclude (prevents duplicate results in same conversation)

    Returns:
        Precise documentation chunks with exact schema definitions, API details, and field specifications
    """
    return await searchAPI(
        query=query,
        max_response_length=max_response_length,
        max_chunks=max_chunks,
        include_references=include_references,
        max_depth=max_depth,
        exclude_chunks=exclude_chunks,
    )


@mcp.tool()
async def research_api(question: str, exclude_chunks: str = "") -> str:
    """
    WHEN TO USE: Complex schema analysis requiring multi-step traversal and synthesis.
    Use for questions that need understanding relationships between multiple schemas,
    resolving schema references, or building complete implementation strategies.

    PERFECT FOR COMPLEX SCHEMA ANALYSIS:
    - Understanding entity hierarchies: "What do I need to create a Sponsored Display campaign?"
      (traverses campaign → ad group → targeting → ad schemas)
    - Schema reference resolution: "What creative types exist in Sponsored Display?"
      (follows schema references across multiple definitions)
    - Multi-API workflows: "How to set up complete product advertising with bidding strategies?"
    - Cross-schema validation: "What fields are shared between Sponsored Products and Sponsored Display?"

    EXAMPLES OF COMPLEX QUESTIONS:
    - "What creative types exist in Sponsored Display?" (requires schema traversal)
    - "Walk me through creating a complete Sponsored Brands campaign with video ads"
    - "What bidding strategies are available across all advertising APIs?"
    - "How do targeting options differ between Sponsored Products and Sponsored Display?"

    DIFFERENCE FROM search_api:
    - search_api: Direct field lookup in known schemas ("What fields are in Campaign schema?")
    - research_api: Multi-schema analysis ("What schemas do I need for a complete campaign setup?")

    INTELLIGENT CAPABILITIES:
    - Automatically searches across multiple related schemas
    - Follows schema references and $ref links
    - Synthesizes information from scattered documentation
    - Provides step-by-step implementation guidance
    - Cross-references related APIs and endpoints

    Args:
        question: Complex research question requiring schema analysis (can be broad -
                 the agent will break it down and traverse necessary schemas automatically)
        exclude_chunks: Comma-separated chunk IDs to exclude (same as search_api - prevents duplicate results)

    Returns:
        Comprehensive analysis with complete schema relationships, implementation steps,
        cross-referenced examples, and synthesized guidance from multiple sources
    """
    return await research_api_question(question, exclude_chunks)


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
