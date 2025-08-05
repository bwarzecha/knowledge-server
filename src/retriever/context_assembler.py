"""Context assembly for Knowledge Retriever."""

import logging
from typing import Any, Dict, List

from ..vector_store.embedding_utils import get_token_count
from .data_classes import Chunk, KnowledgeContext, RetrievalStats

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles primary and referenced chunks into complete context."""

    def __init__(self, config):
        """Initialize with retriever configuration."""
        self.config = config

    def assemble_context(
        self,
        query: str,
        primary_chunks: List[Dict[str, Any]],
        referenced_chunks: List[Dict[str, Any]],
        search_time_ms: float,
        expansion_stats: RetrievalStats,
    ) -> KnowledgeContext:
        """
        Assemble complete knowledge context from chunks.

        Args:
            query: Original query string
            primary_chunks: Direct search results
            referenced_chunks: Auto-resolved dependencies
            search_time_ms: Time spent on semantic search
            expansion_stats: Statistics from reference expansion

        Returns:
            Complete KnowledgeContext object
        """
        # Convert raw chunks to Chunk objects
        primary_chunk_objects = self._convert_to_chunk_objects(
            primary_chunks, "primary_result"
        )

        referenced_chunk_objects = self._convert_to_chunk_objects(
            referenced_chunks, "referenced_dependency"
        )

        # Apply context size limits if needed
        primary_final, referenced_final = self._apply_size_limits(
            primary_chunk_objects, referenced_chunk_objects
        )

        # Calculate total token count
        total_tokens = self._estimate_total_tokens(primary_final + referenced_final)

        # Create final retrieval stats
        final_stats = RetrievalStats(
            search_time_ms=search_time_ms,
            expansion_time_ms=expansion_stats.expansion_time_ms,
            total_time_ms=search_time_ms + expansion_stats.expansion_time_ms,
            primary_count=len(primary_final),
            referenced_count=len(referenced_final),
            depth_reached=expansion_stats.depth_reached,
            circular_refs_detected=expansion_stats.circular_refs_detected,
        )

        context = KnowledgeContext(
            query=query,
            primary_chunks=primary_final,
            referenced_chunks=referenced_final,
            total_chunks=len(primary_final) + len(referenced_final),
            total_tokens=total_tokens,
            retrieval_stats=final_stats,
        )

        logger.info(
            f"Assembled context: {context.total_chunks} chunks, "
            f"{total_tokens} tokens, {final_stats.total_time_ms:.1f}ms total"
        )

        return context

    def _convert_to_chunk_objects(
        self, raw_chunks: List[Dict[str, Any]], retrieval_reason: str
    ) -> List[Chunk]:
        """Convert raw chunks to Chunk objects with relevance scores."""
        chunk_objects = []

        for chunk in raw_chunks:
            # Calculate relevance score
            if retrieval_reason == "primary_result":
                # For primary results, use distance from vector search (lower = more relevant)
                distance = chunk.get("distance", 0.0)
                relevance_score = max(0.0, 1.0 - distance)
            else:
                # For referenced chunks, use fixed lower relevance
                relevance_score = 0.5

            chunk_obj = Chunk(
                id=chunk["id"],
                document=chunk["document"],
                metadata=chunk["metadata"],
                relevance_score=relevance_score,
                retrieval_reason=retrieval_reason,
            )
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def _apply_size_limits(
        self, primary_chunks: List[Chunk], referenced_chunks: List[Chunk]
    ) -> tuple[List[Chunk], List[Chunk]]:
        """Apply chunk and token limits with prioritization strategy."""
        total_chunks = len(primary_chunks) + len(referenced_chunks)

        # If under limits, return all chunks
        if total_chunks <= self.config.max_total_chunks:
            return primary_chunks, referenced_chunks

        # Apply 60/40 prioritization strategy
        if self.config.prioritize_primary:
            primary_limit = min(
                len(primary_chunks), int(self.config.max_total_chunks * 0.6)
            )
            reference_limit = self.config.max_total_chunks - primary_limit
        else:
            # Equal split if not prioritizing primary
            primary_limit = min(len(primary_chunks), self.config.max_total_chunks // 2)
            reference_limit = self.config.max_total_chunks - primary_limit

        # Keep highest relevance chunks within limits
        primary_final = primary_chunks[:primary_limit]

        # Sort referenced chunks by relevance and take top N
        referenced_sorted = sorted(
            referenced_chunks, key=lambda x: x.relevance_score, reverse=True
        )
        referenced_final = referenced_sorted[:reference_limit]

        if total_chunks > self.config.max_total_chunks:
            logger.info(
                f"Applied size limits: kept {len(primary_final)} primary + "
                f"{len(referenced_final)} referenced of {total_chunks} total"
            )

        return primary_final, referenced_final

    def _estimate_total_tokens(self, chunks: List[Chunk]) -> int:
        """Estimate total token count for all chunks."""
        total_tokens = 0

        for chunk in chunks:
            # Count tokens in the document content
            chunk_tokens = get_token_count(chunk.document)
            total_tokens += chunk_tokens

        logger.debug(f"Estimated {total_tokens} tokens across {len(chunks)} chunks")
        return total_tokens
