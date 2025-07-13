"""Reference expansion with circular protection for Knowledge Retriever."""

import logging
import time
from typing import Any, Dict, List, Set, Tuple

from .data_classes import RetrievalStats

logger = logging.getLogger(__name__)


class ReferenceExpander:
    """Expands references using breadth-first traversal with circular protection."""

    def __init__(self, vector_store_manager):
        """Initialize with vector store for chunk lookups."""
        self.vector_store = vector_store_manager

    def expand_references(
        self, primary_chunks: List[Dict[str, Any]], max_depth: int = 3, max_total: int = 15
    ) -> Tuple[List[Dict[str, Any]], RetrievalStats]:
        """
        Expand references from primary chunks using breadth-first traversal.

        Args:
            primary_chunks: Initial chunks to expand from
            max_depth: Maximum reference depth to follow
            max_total: Maximum total referenced chunks to return

        Returns:
            Tuple of (referenced_chunks, expansion_stats)
        """
        start_time = time.time()

        visited_ids: Set[str] = set()
        referenced_chunks: List[Dict[str, Any]] = []
        expansion_queue: List[Tuple[str, int]] = []  # (chunk_id, depth)
        circular_refs_detected = 0
        max_depth_reached = 0

        # Initialize visited set with primary chunk IDs
        for chunk in primary_chunks:
            visited_ids.add(chunk["id"])

        # Collect initial references from primary chunks
        for chunk in primary_chunks:
            ref_ids = chunk["metadata"].get("ref_ids", {})
            for ref_id in ref_ids.keys():
                if ref_id not in visited_ids:
                    expansion_queue.append((ref_id, 1))
                else:
                    circular_refs_detected += 1
                    logger.debug(f"Circular reference detected: {ref_id}")

        # Breadth-first expansion
        while expansion_queue and len(referenced_chunks) < max_total:
            current_id, depth = expansion_queue.pop(0)
            max_depth_reached = max(max_depth_reached, depth)

            # Skip if already visited or depth exceeded
            if current_id in visited_ids:
                circular_refs_detected += 1
                continue

            if depth > max_depth:
                logger.debug(f"Max depth {max_depth} reached, skipping {current_id}")
                continue

            # Retrieve referenced chunk
            ref_chunks = self.vector_store.get_by_ids([current_id])
            if not ref_chunks:
                logger.warning(f"Referenced chunk not found: {current_id}")
                continue

            ref_chunk = ref_chunks[0]
            visited_ids.add(current_id)
            referenced_chunks.append(ref_chunk)

            logger.debug(f"Expanded reference: {current_id} at depth {depth}")

            # Add next level references to queue
            next_refs = ref_chunk["metadata"].get("ref_ids", {})
            for next_ref in next_refs.keys():
                if next_ref not in visited_ids:
                    expansion_queue.append((next_ref, depth + 1))
                else:
                    circular_refs_detected += 1

        expansion_time_ms = (time.time() - start_time) * 1000

        stats = RetrievalStats(
            search_time_ms=0.0,  # Not used in expansion
            expansion_time_ms=expansion_time_ms,
            total_time_ms=expansion_time_ms,
            primary_count=len(primary_chunks),
            referenced_count=len(referenced_chunks),
            depth_reached=max_depth_reached,
            circular_refs_detected=circular_refs_detected,
        )

        logger.info(
            f"Reference expansion completed: {len(referenced_chunks)} chunks, "
            f"depth {max_depth_reached}, {circular_refs_detected} circular refs, "
            f"{expansion_time_ms:.1f}ms"
        )

        return referenced_chunks, stats
