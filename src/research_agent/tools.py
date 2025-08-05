"""Core tools for Research Agent functionality."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse

from ..cli.config import Config
from ..vector_store.vector_store_manager import VectorStoreManager
from .data_classes import (ChunkSummary, FullChunk, RetrievalResults,
                           SearchResults)

logger = logging.getLogger(__name__)


async def searchChunks(
    vector_store: VectorStoreManager,
    api_context: str,
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False,
    rerank: bool = True,
    search_context: Optional[str] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
) -> SearchResults:
    """
    Search for relevant chunks with optional LLM-based relevance filtering.

    Args:
        vector_store: VectorStoreManager instance for search operations
        api_context: Available API files context string
        query: Natural language search query
        max_chunks: Maximum chunks to return (default: 25)
        file_filter: Optional file pattern (e.g., "sponsored-display", "dsp")
        include_references: Whether to include ref_ids in response
        rerank: Whether to apply LLM-based relevance filtering (default: True)
        search_context: Enhanced context about search intent for better filtering

    Returns:
        SearchResults with chunk summaries, metadata, and optional filtering stats
    """
    start_time = time.time()

    # Load configuration for filtering
    config = Config()

    # Calculate search limit based on reranking strategy
    if rerank:
        search_limit = int(max_chunks * config.chunk_filtering_oversample_multiplier)
        logger.info(
            f"Reranking enabled: searching for {search_limit} chunks, filtering to {max_chunks}"
        )
    else:
        search_limit = max_chunks
        logger.info(f"Reranking disabled: searching for {search_limit} chunks directly")

    # Build metadata filters for file filtering
    filters = None
    if file_filter:
        # Get all unique source files and find matches for substring filtering
        # This is a workaround since ChromaDB doesn't support substring matching
        all_chunks = vector_store.collection.get(include=["metadatas"])

        matching_files = set()
        for metadata in all_chunks["metadatas"]:
            source_file = metadata.get("source_file", "")
            if file_filter.lower() in source_file.lower():
                matching_files.add(source_file)

        if matching_files:
            # Use $in operator with exact filenames that match the filter
            filters = {"source_file": {"$in": list(matching_files)}}
        else:
            # No matching files found, search will return empty
            filters = {"source_file": {"$eq": "__no_match__"}}

    # Perform vector search with proper filters
    search_results = vector_store.search(query, limit=search_limit, filters=filters)

    # Convert to ChunkSummary objects
    chunks = []
    files_searched = set()

    for result in search_results:
        # Extract metadata fields
        metadata = result.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")
        chunk_type = metadata.get("type", "unknown")
        natural_name = metadata.get("natural_name", "")

        files_searched.add(source_file)

        # Calculate relevance score from distance (lower distance = higher relevance)
        distance = result.get("distance", 1.0)
        relevance_score = max(0.0, min(1.0, 1.0 - distance))

        # Use full content (no artificial truncation)
        content = result.get("document", "")
        content_preview = content.strip()

        # Create human-readable title
        title = natural_name or f"{source_file}:{result.get('id', 'unknown')}"

        # Handle ref_ids based on include_references
        ref_ids = []
        if include_references:
            ref_ids_data = metadata.get("ref_ids", {})
            if isinstance(ref_ids_data, dict):
                ref_ids = list(ref_ids_data.keys())
            elif isinstance(ref_ids_data, list):
                ref_ids = ref_ids_data

        chunk_summary = ChunkSummary(
            chunk_id=result["id"],
            title=title,
            content_preview=content_preview,
            chunk_type=chunk_type,
            file_name=source_file,
            ref_ids=ref_ids,
            relevance_score=relevance_score,
        )
        chunks.append(chunk_summary)

    # Filter out excluded chunks
    if exclude_chunk_ids:
        exclude_set = set(exclude_chunk_ids)
        original_count = len(chunks)
        chunks = [chunk for chunk in chunks if chunk.chunk_id not in exclude_set]
        excluded_count = original_count - len(chunks)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} chunks from search results")

    # Apply LLM-based filtering if enabled
    filtering_stats = None
    if rerank and chunks:
        try:
            filtered_chunks, filtering_stats = await _filter_and_expand_chunks(
                chunks=chunks,
                query=query,
                search_context=search_context,
                target_count=max_chunks,
                vector_store=vector_store,
                config=config,
            )
            chunks = filtered_chunks
            logger.info(
                f"Applied LLM filtering: {filtering_stats['original_count']} -> "
                f"{filtering_stats['filtered_count']} chunks"
            )
        except Exception as e:
            logger.error(f"LLM filtering failed: {e}. Continuing with original chunks.")
            # Use fallback: just trim to max_chunks
            chunks = chunks[:max_chunks]
    elif not rerank:
        # No reranking: just trim to max_chunks if needed
        chunks = chunks[:max_chunks]

    total_time_ms = (time.time() - start_time) * 1000

    return SearchResults(
        chunks=chunks,
        total_found=len(chunks),
        search_time_ms=total_time_ms,
        files_searched=list(files_searched),
        api_context=api_context,
        filtering_stats=filtering_stats,
    )


async def getChunks(
    vector_store: VectorStoreManager,
    chunk_ids: List[str],
    expand_depth: int = 3,
    max_total_chunks: int = 100,
) -> RetrievalResults:
    """
    Retrieve multiple chunks by ID with controlled reference expansion.

    Args:
        vector_store: VectorStoreManager instance for retrieval operations
        chunk_ids: List of chunk IDs to retrieve
        expand_depth: Reference expansion depth (0-10+, default: 3)
        max_total_chunks: Limit total chunks including expansions (default: 100)

    Returns:
        RetrievalResults with full chunk content and expansion details
    """
    # Handle empty input
    if not chunk_ids:
        return RetrievalResults(
            requested_chunks=[],
            expanded_chunks=[],
            total_chunks=0,
            total_tokens=0,
            expansion_stats={},
            truncated=False,
        )

    # Get requested chunks by ID
    chunk_results = vector_store.get_by_ids(chunk_ids)

    # Convert to FullChunk objects
    requested_chunks = []
    total_tokens = 0

    for result in chunk_results:
        content = result.get("document", "")
        metadata = result.get("metadata", {})

        # Calculate tokens (rough estimation: 4 chars per token)
        chunk_tokens = len(content) // 4
        total_tokens += chunk_tokens

        # Extract ref_ids from metadata
        ref_ids_data = metadata.get("ref_ids", {})
        if isinstance(ref_ids_data, dict):
            ref_ids = list(ref_ids_data.keys())
        elif isinstance(ref_ids_data, list):
            ref_ids = ref_ids_data
        else:
            ref_ids = []

        full_chunk = FullChunk(
            chunk_id=result["id"],
            content=content,
            metadata=metadata,
            source="requested",
            expansion_depth=0,
            ref_ids=ref_ids,
        )
        requested_chunks.append(full_chunk)

    # Initialize expansion tracking
    expanded_chunks = []
    expansion_stats = {0: len(requested_chunks)}

    # Reference expansion logic for expand_depth > 0
    if expand_depth > 0:
        visited_ids = {chunk.chunk_id for chunk in requested_chunks}
        expansion_queue = []  # [(chunk_id, depth)]

        # Collect initial references from requested chunks
        for chunk in requested_chunks:
            for ref_id in chunk.ref_ids:
                if ref_id not in visited_ids:
                    expansion_queue.append((ref_id, 1))

        # Breadth-first expansion
        while (
            expansion_queue
            and len(requested_chunks) + len(expanded_chunks) < max_total_chunks
        ):
            current_id, depth = expansion_queue.pop(0)

            # Skip if already visited or depth exceeded
            if current_id in visited_ids or depth > expand_depth:
                continue

            # Retrieve referenced chunk
            ref_chunks = vector_store.get_by_ids([current_id])
            if not ref_chunks:
                continue

            ref_result = ref_chunks[0]
            content = ref_result.get("document", "")
            metadata = ref_result.get("metadata", {})

            # Calculate tokens for this chunk
            chunk_tokens = len(content) // 4
            total_tokens += chunk_tokens

            # Extract ref_ids from metadata
            ref_ids_data = metadata.get("ref_ids", {})
            if isinstance(ref_ids_data, dict):
                ref_ids = list(ref_ids_data.keys())
            elif isinstance(ref_ids_data, list):
                ref_ids = ref_ids_data
            else:
                ref_ids = []

            # Create expanded chunk
            expanded_chunk = FullChunk(
                chunk_id=current_id,
                content=content,
                metadata=metadata,
                source="expanded",
                expansion_depth=depth,
                ref_ids=ref_ids,
            )
            expanded_chunks.append(expanded_chunk)
            visited_ids.add(current_id)

            # Update expansion stats
            if depth not in expansion_stats:
                expansion_stats[depth] = 0
            expansion_stats[depth] += 1

            # Add next level references to queue
            for next_ref in ref_ids:
                if next_ref not in visited_ids and depth < expand_depth:
                    expansion_queue.append((next_ref, depth + 1))

    # Check if expansion was truncated
    truncated = len(requested_chunks) + len(expanded_chunks) >= max_total_chunks

    return RetrievalResults(
        requested_chunks=requested_chunks,
        expanded_chunks=expanded_chunks,
        total_chunks=len(requested_chunks) + len(expanded_chunks),
        total_tokens=total_tokens,
        expansion_stats=expansion_stats,
        truncated=truncated,
    )


def generate_api_context(api_index_path: str = "data/api_index.json") -> str:
    """
    Generate API files context from api_index.json.

    Args:
        api_index_path: Path to api_index.json file

    Returns:
        Formatted API context string
    """
    try:
        index_path = Path(api_index_path)
        if not index_path.exists():
            return "Available API Files: (context unavailable)"

        with open(index_path, "r") as f:
            api_index = json.load(f)

        # Handle different API index formats
        if "files" in api_index:
            # New format with files array
            files = api_index["files"]
            lines = ["Available API Files:"]
            for file_info in files:
                file_name = file_info["file"]
                # Extract description from text field if available
                description = (
                    file_info.get("text", "").split("\n")[0]
                    if file_info.get("text")
                    else "API Documentation"
                )
                lines.append(f"- {file_name}: {description}")
        else:
            # Old format with direct file mapping
            lines = ["Available API Files:"]
            for file_name, file_info in api_index.items():
                if isinstance(file_info, dict) and "description" in file_info:
                    description = file_info["description"]
                    lines.append(f"- {file_name}: {description}")
                else:
                    lines.append(f"- {file_name}: API Documentation")

        return "\n".join(lines)

    except Exception:
        return "Available API Files: (context unavailable)"


def _build_intelligence_prompt(
    query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int
) -> str:
    """Build prompt for LLM relevance assessment and expansion decisions."""

    # Enhanced context for better filtering decisions
    context_section = ""
    if search_context:
        context_section = f"""
Search Context: {search_context}

This context explains WHY this search was initiated and WHAT the user is trying to accomplish.
Use this context to make better relevance decisions.
"""

    prompt = f"""You are an expert API documentation analyst. \
Evaluate chunks for relevance to the user's specific needs and decide appropriate expansion strategy.

User Query: "{query}"{context_section}

For each relevant chunk, decide the best action to provide comprehensive but focused context.

Chunks to evaluate:
"""

    for chunk in chunk_info:
        prompt += f"""
Chunk ID: {chunk["id"]}
Title: {chunk["title"]}
Type: {chunk["type"]}
File: {chunk["file"]}
References: {chunk["ref_ids"]}
Vector Score: {chunk["relevance_score"]:.3f}
Content:
{chunk["content"]}

---
"""

    prompt += f"""
Instructions:
1. Evaluate each chunk for relevance to the user's query and context
2. For relevant chunks, decide the appropriate action based on content and references:
   - KEEP: Chunk is complete and directly answers the query
   - EXPAND_1: Needs basic context from immediate references
   - EXPAND_3: Needs moderate expansion for schemas or related concepts
   - EXPAND_5: Needs deep expansion for complex nested structures
   - DISCARD: Not relevant to the query or context
3. Consider chunk type, references, and search context when deciding expansion depth
4. Aim for comprehensive but focused results (target ~{target_count} relevant chunks)

Response format: One decision per line
chunk_id -> action

Example:
sd-api:CreateCampaign -> EXPAND_3
dsp-api:GetAudiences -> KEEP
sd-api:ErrorCodes -> DISCARD

Note: Any chunk not mentioned is automatically discarded
"""

    return prompt


def _parse_relevance_response(response: str) -> Dict[str, str]:
    """Parse LLM response to extract chunk decisions by ID."""
    try:
        # Parse line-by-line "chunk_id -> action" format (handles spaces around ->)
        decisions = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            if "->" in line and line:
                # Split on arrow separator, handling optional spaces
                parts = line.split("->", 1)
                if len(parts) == 2:
                    chunk_id, action = parts
                    decisions[chunk_id.strip()] = action.strip()

        logger.info(f"Parsed {len(decisions)} chunk decisions from LLM response")
        return decisions
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {response}. Error: {e}")
        return {}  # Return empty dict to trigger fallback


async def _filter_and_expand_chunks(
    chunks: List[ChunkSummary],
    query: str,
    search_context: Optional[str],
    target_count: int,
    vector_store: VectorStoreManager,
    config: Config,
) -> Tuple[List[ChunkSummary], Dict[str, Any]]:
    """
    Use LLM to filter chunks and make intelligent expansion decisions.

    Args:
        chunks: List of chunk summaries from vector search
        query: Original user query
        search_context: Enhanced context about the search intent
        target_count: Desired number of chunks after filtering
        vector_store: For chunk lookups and validation
        config: Configuration for LLM settings

    Returns:
        Tuple of (filtered_chunks, processing_stats)
    """
    start_time = time.time()

    # Prepare chunk information for LLM with full content and references
    chunk_info = []
    for chunk in chunks:
        chunk_info.append(
            {
                "id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content_preview,  # Full content (no truncation)
                "type": chunk.chunk_type,
                "file": chunk.file_name,
                "ref_ids": chunk.ref_ids,
                "relevance_score": chunk.relevance_score,
            }
        )

    # Build LLM prompt with enhanced context
    prompt = _build_intelligence_prompt(query, search_context, chunk_info, target_count)

    try:
        # Configure retry behavior at AWS SDK level for reranker
        aws_config = BotocoreConfig(
            retries={
                "max_attempts": config.chunk_filtering_llm_retry_max_attempts,
                "mode": "standard",
            }
        )

        # Create dedicated re-ranker LLM model with separate config
        model = ChatBedrockConverse(
            model=config.reranker_llm_model,
            temperature=config.reranker_llm_temperature,
            max_tokens=config.reranker_llm_max_tokens,
            config=aws_config,
        )

        # Call LLM for relevance assessment
        logger.info(
            f"Calling LLM to filter {len(chunks)} chunks for query: {query[:50]}..."
        )
        response = await model.ainvoke([{"role": "user", "content": prompt}])

        # Parse LLM response to get chunk decisions by ID
        chunk_decisions = _parse_relevance_response(response.content)

        if not chunk_decisions:
            raise ValueError("No valid decisions parsed from LLM response")

        # Filter chunks based on LLM decisions
        filtered_chunks = []
        for chunk in chunks:
            action = chunk_decisions.get(chunk.chunk_id, "DISCARD")
            if action != "DISCARD":
                filtered_chunks.append(chunk)

        filtering_time = (time.time() - start_time) * 1000

        stats = {
            "llm_intelligence_enabled": True,
            "original_count": len(chunks),
            "filtered_count": len(filtered_chunks),
            "reduction_ratio": (
                1.0 - (len(filtered_chunks) / len(chunks)) if chunks else 0.0
            ),
            "decisions": chunk_decisions,
            "processing_time_ms": filtering_time,
            "fallback_used": False,
        }

        logger.info(
            f"LLM filtering: {len(chunks)} -> {len(filtered_chunks)} chunks "
            f"({stats['reduction_ratio']:.1%} reduction) in {filtering_time:.1f}ms"
        )

        return filtered_chunks, stats

    except Exception as e:
        # Fallback: return top chunks by vector similarity
        logger.warning(f"LLM intelligence failed: {e}. Using fallback.")
        fallback_chunks = chunks[:target_count]

        stats = {
            "llm_intelligence_enabled": True,
            "original_count": len(chunks),
            "filtered_count": len(fallback_chunks),
            "reduction_ratio": (
                1.0 - (len(fallback_chunks) / len(chunks)) if chunks else 0.0
            ),
            "decisions": {},
            "processing_time_ms": 0,
            "fallback_used": True,
            "error": str(e),
        }

        return fallback_chunks, stats
