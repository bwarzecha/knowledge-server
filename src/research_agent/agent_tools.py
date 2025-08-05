"""LangChain tool wrappers for Research Agent."""

import logging
from typing import List, Optional

from langchain_core.tools import tool

from ..mcp_server.shared_resources import get_shared_resources
from .tools import generate_api_context, getChunks, searchChunks

logger = logging.getLogger(__name__)


@tool
async def search_chunks_tool(
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False,
    rerank: bool = True,
    search_context: Optional[str] = None,
    exclude_chunks: str = "",
) -> dict:
    """Search API documentation chunks with intelligent LLM-based relevance filtering.

    Use this tool to find relevant documentation chunks. The rerank parameter enables
    intelligent filtering that dramatically improves result quality by analyzing chunk
    relevance using LLM understanding.

    IMPORTANT: When rerank=True (default), provide rich search_context to help the LLM
    make better filtering decisions. Include:
    - WHY you're searching (user's goal)
    - WHAT you're trying to accomplish
    - Any previous findings or context

    Args:
        query: Search query text
        max_chunks: Maximum chunks to return after filtering
        file_filter: Optional API filter ("sponsored-display", "dsp", etc.)
        include_references: Include reference IDs in response
        rerank: Enable LLM filtering for better results (default: True)
        search_context: Rich context about search intent for quality filtering
        exclude_chunks: Comma-separated chunk IDs to exclude from results

    Examples:
        # Basic search
        search_chunks_tool("authentication", max_chunks=5)

        # With rich context for better filtering
        search_chunks_tool(
            query="campaign optimization",
            search_context=(
                "User wants to understand how to configure automatic bid "
                "optimization rules and performance targets for advertising campaigns"
            ),
            max_chunks=5
        )
    """
    logger.info(
        f"search_chunks_tool called: query='{query}', max_chunks={max_chunks}, "
        f"file_filter={file_filter}, rerank={rerank}"
    )

    # Parse exclude_chunks parameter
    exclude_chunk_ids = []
    if exclude_chunks.strip():
        exclude_chunk_ids = [
            chunk_id.strip()
            for chunk_id in exclude_chunks.split(",")
            if chunk_id.strip()
        ]

    resources = get_shared_resources()
    api_context = generate_api_context()

    result = await searchChunks(
        vector_store=resources.vector_store,
        api_context=api_context,
        query=query,
        max_chunks=max_chunks,
        file_filter=file_filter,
        include_references=include_references,
        rerank=rerank,
        search_context=search_context,
        exclude_chunk_ids=exclude_chunk_ids,
    )

    logger.info(
        f"search_chunks_tool result: {result.total_found} chunks found in {result.search_time_ms:.1f}ms"
    )

    # Log filtering stats if available
    if result.filtering_stats:
        stats = result.filtering_stats
        logger.info(
            f"LLM filtering: {stats['original_count']} → {stats['filtered_count']} chunks "
            f"({stats['reduction_ratio']:.1%} reduction), "
            f"processing_time={stats['processing_time_ms']:.0f}ms, "
            f"fallback_used={stats['fallback_used']}"
        )

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
async def get_chunks_tool(
    chunk_ids: List[str], expand_depth: int = 3, max_total_chunks: int = 100
) -> dict:
    """Retrieve specific chunks by ID with reference expansion.

    Use expand_depth=0 for quick lookups, expand_depth=3-5 for complete schemas,
    expand_depth=5+ for deep nested structures.
    """
    logger.info(
        f"get_chunks_tool called: {len(chunk_ids)} chunk_ids, expand_depth={expand_depth}"
    )

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
