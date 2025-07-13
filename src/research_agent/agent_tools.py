"""LangChain tool wrappers for Research Agent."""

import logging
from typing import List, Optional

from langchain_core.tools import tool

from ..mcp_server.shared_resources import get_shared_resources
from .tools import generate_api_context, getChunks, searchChunks

logger = logging.getLogger(__name__)


@tool
async def search_chunks_tool(
    query: str, max_chunks: int = 25, file_filter: Optional[str] = None, include_references: bool = False
) -> dict:
    """Search API documentation chunks with optional file filtering.

    Use this tool to find relevant documentation chunks. Use file_filter when user mentions
    specific APIs (e.g., "sponsored-display", "dsp", "seller-central").
    """
    logger.info(f"search_chunks_tool called: query='{query}', max_chunks={max_chunks}, file_filter={file_filter}")

    resources = get_shared_resources()
    api_context = generate_api_context()

    result = await searchChunks(
        vector_store=resources.vector_store,
        api_context=api_context,
        query=query,
        max_chunks=max_chunks,
        file_filter=file_filter,
        include_references=include_references,
    )

    logger.info(f"search_chunks_tool result: {result.total_found} chunks found in {result.search_time_ms:.1f}ms")

    # Convert to dict for LLM consumption
    return {
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


@tool
async def get_chunks_tool(chunk_ids: List[str], expand_depth: int = 3, max_total_chunks: int = 100) -> dict:
    """Retrieve specific chunks by ID with reference expansion.

    Use expand_depth=0 for quick lookups, expand_depth=3-5 for complete schemas,
    expand_depth=5+ for deep nested structures.
    """
    logger.info(f"get_chunks_tool called: {len(chunk_ids)} chunk_ids, expand_depth={expand_depth}")

    resources = get_shared_resources()

    result = await getChunks(
        vector_store=resources.vector_store,
        chunk_ids=chunk_ids,
        expand_depth=expand_depth,
        max_total_chunks=max_total_chunks,
    )

    logger.info(
        f"get_chunks_tool result: {result.total_chunks} total chunks "
        f"({len(result.requested_chunks)} requested, {len(result.expanded_chunks)} expanded)"
    )

    # Convert to dict for LLM consumption
    return {
        "requested_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "expansion_depth": chunk.expansion_depth,
                "ref_ids": chunk.ref_ids,
            }
            for chunk in result.requested_chunks
        ],
        "expanded_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "expansion_depth": chunk.expansion_depth,
                "ref_ids": chunk.ref_ids,
            }
            for chunk in result.expanded_chunks
        ],
        "total_chunks": result.total_chunks,
        "total_tokens": result.total_tokens,
        "expansion_stats": result.expansion_stats,
        "truncated": result.truncated,
    }

