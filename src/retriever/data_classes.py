"""Data classes for Knowledge Retriever component."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    """Individual chunk with retrieval metadata."""

    id: str
    document: str
    metadata: Dict[str, Any]
    relevance_score: float
    retrieval_reason: str  # "primary_result" or "referenced_dependency"


@dataclass
class RetrievalStats:
    """Performance metrics for retrieval operation."""

    search_time_ms: float
    expansion_time_ms: float
    total_time_ms: float
    primary_count: int
    referenced_count: int
    depth_reached: int
    circular_refs_detected: int


@dataclass
class KnowledgeContext:
    """Complete retrieval results for a query."""

    query: str
    primary_chunks: List[Chunk]
    referenced_chunks: List[Chunk]
    total_chunks: int
    total_tokens: int
    retrieval_stats: RetrievalStats


@dataclass
class RetrieverConfig:
    """Configuration for Knowledge Retriever."""

    max_primary_results: int = 5
    max_total_chunks: int = 15
    max_depth: int = 3
    timeout_ms: int = 5000
    token_limit: int = 4000
    prioritize_primary: bool = True

    @classmethod
    def from_env(cls) -> "RetrieverConfig":
        """Create configuration from environment variables."""
        return cls(
            max_primary_results=int(os.getenv("RETRIEVAL_MAX_PRIMARY_RESULTS", "5")),
            max_total_chunks=int(os.getenv("RETRIEVAL_MAX_TOTAL_CHUNKS", "15")),
            max_depth=int(os.getenv("RETRIEVAL_MAX_DEPTH", "3")),
            timeout_ms=int(os.getenv("RETRIEVAL_TIMEOUT_MS", "5000")),
            token_limit=int(os.getenv("CONTEXT_TOKEN_LIMIT", "4000")),
            prioritize_primary=os.getenv("CONTEXT_PRIORITIZE_PRIMARY", "true").lower() == "true",
        )
