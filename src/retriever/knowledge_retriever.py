"""Main Knowledge Retriever orchestrator."""

import logging
import time
from typing import Any, Dict, Optional

from .context_assembler import ContextAssembler
from .data_classes import KnowledgeContext, RetrieverConfig
from .reference_expander import ReferenceExpander

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Main orchestrator for intelligent knowledge retrieval."""

    def __init__(self, vector_store_manager, config: Optional[RetrieverConfig] = None):
        """
        Initialize Knowledge Retriever.

        Args:
            vector_store_manager: Vector Store Manager instance for search and lookup
            config: Optional configuration, defaults to environment-based config
        """
        self.vector_store = vector_store_manager
        self.config = config or RetrieverConfig.from_env()

        # Initialize sub-components
        self.reference_expander = ReferenceExpander(vector_store_manager)
        self.context_assembler = ContextAssembler(self.config)

        logger.info(
            f"Initialized KnowledgeRetriever with config: "
            f"max_primary={self.config.max_primary_results}, "
            f"max_total={self.config.max_total_chunks}, "
            f"max_depth={self.config.max_depth}"
        )

    def retrieve_knowledge(
        self,
        query: str,
        max_primary_results: Optional[int] = None,
        max_total_chunks: Optional[int] = None,
        include_references: bool = True,
        max_depth: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeContext:
        """
        Main retrieval interface implementing two-stage pipeline.

        Args:
            query: Natural language query
            max_primary_results: Override config for primary results limit
            max_total_chunks: Override config for total chunks limit
            include_references: Whether to expand references (default: True)
            max_depth: Override config for reference expansion depth
            filters: Optional metadata filters for initial search

        Returns:
            Complete KnowledgeContext with primary and referenced chunks
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return self._create_empty_context(query)

        # Use provided parameters or fall back to config
        primary_limit = max_primary_results or self.config.max_primary_results
        total_limit = max_total_chunks or self.config.max_total_chunks
        depth_limit = max_depth or self.config.max_depth

        logger.info(
            f"Starting retrieval for query: '{query}' (primary_limit={primary_limit}, total_limit={total_limit})"
        )

        # Stage 1: Semantic Search
        search_start = time.time()
        primary_chunks = self._execute_semantic_search(query, primary_limit, filters)
        search_time_ms = (time.time() - search_start) * 1000

        if not primary_chunks:
            logger.info("No primary chunks found for query")
            return self._create_empty_context(query)

        logger.info(f"Found {len(primary_chunks)} primary chunks in {search_time_ms:.1f}ms")

        # Stage 2: Reference Expansion (if enabled)
        referenced_chunks = []
        expansion_stats = None

        if include_references:
            max_references = total_limit - len(primary_chunks)
            if max_references > 0:
                referenced_chunks, expansion_stats = self.reference_expander.expand_references(
                    primary_chunks=primary_chunks,
                    max_depth=depth_limit,
                    max_total=max_references,
                )
                logger.info(f"Expanded to {len(referenced_chunks)} referenced chunks")
            else:
                logger.info("Skipping reference expansion - primary chunks at limit")
        else:
            logger.info("Reference expansion disabled")

        # Create empty expansion stats if not generated
        if expansion_stats is None:
            from .data_classes import RetrievalStats

            expansion_stats = RetrievalStats(
                search_time_ms=0,
                expansion_time_ms=0,
                total_time_ms=0,
                primary_count=len(primary_chunks),
                referenced_count=0,
                depth_reached=0,
                circular_refs_detected=0,
            )

        # Stage 3: Context Assembly
        context = self.context_assembler.assemble_context(
            query=query,
            primary_chunks=primary_chunks,
            referenced_chunks=referenced_chunks,
            search_time_ms=search_time_ms,
            expansion_stats=expansion_stats,
        )

        logger.info(
            f"Retrieval completed: {context.total_chunks} chunks, "
            f"{context.total_tokens} tokens, {context.retrieval_stats.total_time_ms:.1f}ms"
        )

        return context

    def _execute_semantic_search(self, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> list:
        """Execute semantic search using Vector Store Manager."""
        try:
            results = self.vector_store.search(query=query, limit=limit, filters=filters)

            logger.debug(f"Semantic search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _create_empty_context(self, query: str) -> KnowledgeContext:
        """Create empty but valid KnowledgeContext for edge cases."""
        from .data_classes import RetrievalStats

        empty_stats = RetrievalStats(
            search_time_ms=0,
            expansion_time_ms=0,
            total_time_ms=0,
            primary_count=0,
            referenced_count=0,
            depth_reached=0,
            circular_refs_detected=0,
        )

        return KnowledgeContext(
            query=query,
            primary_chunks=[],
            referenced_chunks=[],
            total_chunks=0,
            total_tokens=0,
            retrieval_stats=empty_stats,
        )

    def get_config(self) -> RetrieverConfig:
        """Get current configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
