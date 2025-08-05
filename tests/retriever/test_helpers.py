"""Test helpers for Knowledge Retriever tests."""

from typing import Any, Dict, List

from src.cli.config import Config
from src.openapi_processor.processor import OpenAPIProcessor


def get_real_test_chunks() -> List[Dict[str, Any]]:
    """Get real chunks from openapi-workshop samples for testing."""
    config = Config()
    processor = OpenAPIProcessor(config)
    chunks = processor.process_directory(
        "open-api-small-samples/3.0/json/openapi-workshop"
    )
    return chunks


def get_chunks_with_references() -> List[Dict[str, Any]]:
    """Get only chunks that have references for expansion testing."""
    chunks = get_real_test_chunks()
    return [chunk for chunk in chunks if chunk["metadata"].get("ref_ids")]


def get_chunks_by_type(chunk_type: str) -> List[Dict[str, Any]]:
    """Get chunks of specific type (operation, component, info)."""
    chunks = get_real_test_chunks()
    return [chunk for chunk in chunks if chunk["metadata"].get("type") == chunk_type]


def find_chunk_by_id(chunk_id: str) -> Dict[str, Any]:
    """Find specific chunk by ID."""
    chunks = get_real_test_chunks()
    for chunk in chunks:
        if chunk["id"] == chunk_id:
            return chunk
    raise ValueError(f"Chunk not found: {chunk_id}")


def get_circular_reference_chunks() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Get a pair of chunks that reference each other (circular)."""
    chunks = get_real_test_chunks()

    # Find operation and its error chunk that reference each other
    operation_chunk = None
    error_chunk = None

    for chunk in chunks:
        if chunk["id"] == "anyOf-allOf.json:paths/hoot/{id}/get":
            operation_chunk = chunk
        elif chunk["id"] == "anyOf-allOf.json:paths/hoot/{id}/get:errors":
            error_chunk = chunk

    if operation_chunk and error_chunk:
        return operation_chunk, error_chunk

    raise ValueError("Could not find circular reference chunks")


def get_deep_reference_chain() -> List[Dict[str, Any]]:
    """Get chunks that form a deep reference chain for depth testing."""
    chunks = get_real_test_chunks()

    # Find chunks that reference each other in a chain
    # Start with an operation that references schemas
    chain_chunks = []

    for chunk in chunks:
        # Look for operation chunks with schema references
        if chunk["metadata"].get("type") == "operation" and chunk["metadata"].get(
            "ref_ids"
        ):
            chain_chunks.append(chunk)

            # Add the referenced chunks to the chain
            ref_ids = chunk["metadata"]["ref_ids"]
            for ref_id in ref_ids.keys():
                try:
                    ref_chunk = find_chunk_by_id(ref_id)
                    if ref_chunk not in chain_chunks:
                        chain_chunks.append(ref_chunk)
                except ValueError:
                    continue  # Skip missing chunks

            # Return first valid chain found
            if len(chain_chunks) > 1:
                return chain_chunks

    # Fallback: return any chunks with references
    return get_chunks_with_references()[:3]


class MockVectorStore:
    """Mock Vector Store Manager for testing retriever components."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        """Initialize with a list of chunks to serve."""
        self.chunks = chunks
        self.chunks_by_id = {chunk["id"]: chunk for chunk in chunks}

    def search(self, query: str, limit: int = 5, filters=None) -> List[Dict[str, Any]]:
        """Mock search that returns chunks filtered by query keywords."""
        results = []
        query_lower = query.lower()

        for chunk in self.chunks:
            # Simple keyword matching in document content
            if any(word in chunk["document"].lower() for word in query_lower.split()):
                # Add mock distance score
                chunk_with_distance = chunk.copy()
                chunk_with_distance["distance"] = (
                    0.1  # Mock low distance (high relevance)
                )
                results.append(chunk_with_distance)

                if len(results) >= limit:
                    break

        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    if result["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results

        return results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Mock get_by_ids that returns chunks matching the IDs."""
        results = []
        for chunk_id in ids:
            if chunk_id in self.chunks_by_id:
                results.append(self.chunks_by_id[chunk_id])
        return results
