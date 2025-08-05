"""searchAPI tool implementation for MCP server."""

from .response_assembler import ResponseAssembler
from .shared_resources import get_shared_resources


async def searchAPI(
    query: str,
    max_response_length: int = 4000,
    max_chunks: int = 50,
    include_references: bool = True,
    max_depth: int = 3,
    exclude_chunks: str = "",
) -> str:
    """
    Search API documentation and return relevant chunks.

    Args:
        query: Natural language search query for API documentation
        max_response_length: Maximum response length in tokens (for LLM context window)
        max_chunks: Maximum number of chunks to retrieve from knowledge base
        include_references: Whether to follow references between chunks
        max_depth: Maximum depth for reference expansion
        exclude_chunks: Comma-separated chunk IDs to exclude from results

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

        # Parse exclude_chunks parameter
        exclude_chunk_ids = []
        if exclude_chunks.strip():
            exclude_chunk_ids = [
                chunk_id.strip()
                for chunk_id in exclude_chunks.split(",")
                if chunk_id.strip()
            ]

        # Retrieve priority-ordered chunks from knowledge base
        context = resources.retriever.retrieve_knowledge(
            query=query,
            max_total_chunks=max_chunks,
            include_references=include_references,
            max_depth=max_depth,
            exclude_chunk_ids=exclude_chunk_ids,
        )

        # Determine chunk limit mode based on request
        # Use count_based when requesting many chunks (>20) to ensure user gets what they asked for
        chunk_limit_mode = "count_based" if max_chunks > 20 else "token_based"

        # Assemble response with length constraints
        assembler = ResponseAssembler()
        response = assembler.assemble_response(
            context=context,
            max_response_length=max_response_length,
            format_style="detailed",
            chunk_limit_mode=chunk_limit_mode,
        )

        return response

    except Exception as e:
        return f"""Error processing query: "{query}"

Technical details: {str(e)}

---
Confidence: error
Sources: 0 chunks
Response time: error"""
