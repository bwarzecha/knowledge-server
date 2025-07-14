"""Data classes for Research Agent tools."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChunkSummary:
    """Summary of a chunk for search results."""

    chunk_id: str
    title: str
    content_preview: str
    chunk_type: str  # "operation", "component", "schema", "info", "unknown"
    file_name: str
    ref_ids: List[str]
    relevance_score: float


@dataclass
class SearchResults:
    """Results from searchChunks operation."""

    chunks: List[ChunkSummary]
    total_found: int
    search_time_ms: float
    files_searched: List[str]
    api_context: str
    filtering_stats: Optional[Dict[str, Any]] = None


@dataclass
class FullChunk:
    """Complete chunk with full content and metadata."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    source: str  # "requested" or "expanded"
    expansion_depth: int
    ref_ids: List[str]


@dataclass
class RetrievalResults:
    """Results from getChunks operation with expansion."""

    requested_chunks: List[FullChunk]
    expanded_chunks: List[FullChunk]
    total_chunks: int
    total_tokens: int
    expansion_stats: Dict[int, int]  # depth -> chunk count
    truncated: bool
