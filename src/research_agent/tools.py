"""Core tools for Research Agent functionality."""

import json
import time
from pathlib import Path
from typing import List, Optional

from ..vector_store.vector_store_manager import VectorStoreManager
from .data_classes import ChunkSummary, FullChunk, RetrievalResults, SearchResults


async def searchChunks(
    vector_store: VectorStoreManager,
    api_context: str,
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False,
) -> SearchResults:
    """
    Search for relevant chunks with optional file filtering.

    Args:
        vector_store: VectorStoreManager instance for search operations
        api_context: Available API files context string
        query: Natural language search query
        max_chunks: Maximum chunks to return (default: 25)
        file_filter: Optional file pattern (e.g., "sponsored-display", "dsp")
        include_references: Whether to include ref_ids in response

    Returns:
        SearchResults with chunk summaries and metadata
    """
    start_time = time.time()

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
    search_results = vector_store.search(query, limit=max_chunks, filters=filters)
    search_time_ms = (time.time() - start_time) * 1000

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

        # Create content preview (first 200 chars)
        content = result.get("document", "")
        content_preview = content[:200].strip()
        if len(content) > 200:
            content_preview += "..."

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

    return SearchResults(
        chunks=chunks,
        total_found=len(chunks),
        search_time_ms=search_time_ms,
        files_searched=list(files_searched),
        api_context=api_context,
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
        while expansion_queue and len(requested_chunks) + len(expanded_chunks) < max_total_chunks:
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
                description = file_info.get("text", "").split("\n")[0] if file_info.get("text") else "API Documentation"
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
