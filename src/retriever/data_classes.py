"""Data classes for Knowledge Retriever component."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from ..cli.config import Config


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
    def from_config(cls, config: "Config") -> "RetrieverConfig":
        """Create configuration from Config object."""
        return cls(
            max_primary_results=config.retrieval_max_primary_results,
            max_total_chunks=config.retrieval_max_total_chunks,
            max_depth=config.retrieval_max_depth,
            timeout_ms=config.retrieval_timeout_ms,
            token_limit=config.context_token_limit,
            prioritize_primary=config.context_prioritize_primary,
        )
