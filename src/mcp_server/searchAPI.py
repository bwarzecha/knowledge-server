"""searchAPI tool implementation for MCP server."""

from .response_assembler import ResponseAssembler
from .shared_resources import get_shared_resources


async def searchAPI(
    query: str,
    max_response_length: int = 4000,
    max_chunks: int = 50,
    include_references: bool = True,
    max_depth: int = 3,
) -> str:
    """
    Search API documentation and return relevant chunks.

    Args:
        query: Natural language search query for API documentation
        max_response_length: Maximum response length in tokens (for LLM context window)
        max_chunks: Maximum number of chunks to retrieve from knowledge base
        include_references: Whether to follow references between chunks
        max_depth: Maximum depth for reference expansion

    Returns:
        Formatted chunks trimmed to fit within max_response_length
    """
    try:
        # Get shared resources (pre-built indices)
        resources = get_shared_resources()

        if not resources.is_ready():
            return """Error: Server not properly initialized.

The MCP server needs to be started with 'knowledge-server serve' to load pre-built indices.

---
Confidence: error
Sources: 0 chunks
Status: not initialized"""

        # Retrieve priority-ordered chunks from knowledge base
        context = resources.retriever.retrieve_knowledge(
            query=query,
            max_total_chunks=max_chunks,
            include_references=include_references,
            max_depth=max_depth,
        )

        # Assemble response with length constraints
        assembler = ResponseAssembler()
        response = assembler.assemble_response(
            context=context, max_response_length=max_response_length, format_style="detailed"
        )

        return response

    except Exception as e:
        return f"""Error processing query: "{query}"

Technical details: {str(e)}

---
Confidence: error
Sources: 0 chunks
Response time: error"""
