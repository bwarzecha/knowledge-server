"""Integration tests for the complete Vector Store pipeline."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.openapi_processor.processor import OpenAPIProcessor
from src.vector_store.vector_store_manager import VectorStoreManager


class TestVectorStoreIntegration:
    """Integration tests for OpenAPI processing + Vector Store workflow."""

    @pytest.fixture(scope="class")
    def temp_db_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def openapi_chunks(self):
        """Process OpenAPI files and return chunks."""
        processor = OpenAPIProcessor()
        chunks = processor.process_directory("samples")
        # Use subset for testing to keep it manageable
        return chunks[:100]  # First 100 chunks

    @pytest.fixture
    def vector_store(self, temp_db_dir):
        """Create VectorStoreManager for testing."""
        return VectorStoreManager(
            persist_directory=str(Path(temp_db_dir) / "chromadb"),
            collection_name="integration_test",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="cpu",  # Use CPU for test reliability
            max_tokens=256,
            reset_on_start=True,
        )

    def test_full_pipeline_workflow(self, vector_store, openapi_chunks):
        """Test complete workflow: setup -> index -> search -> retrieve."""
        # Setup vector store
        vector_store.setup()

        # Verify initial state
        info = vector_store.get_info()
        assert info["count"] == 0

        # Index chunks
        vector_store.add_chunks(openapi_chunks, batch_size=25)

        # Verify indexing
        info = vector_store.get_info()
        assert info["count"] == len(openapi_chunks)

        # Test search functionality
        search_results = vector_store.search("API endpoint for user management", limit=5)
        assert len(search_results) <= 5
        assert all("id" in result for result in search_results)
        assert all("distance" in result for result in search_results)

        # Test filtered search
        component_results = vector_store.search("schema definition", limit=3, filters={"type": "component"})

        # Verify all results are components (if any found)
        for result in component_results:
            assert result["metadata"]["type"] == "component"

        # Test ID retrieval
        if search_results:
            test_ids = [search_results[0]["id"]]
            retrieved = vector_store.get_by_ids(test_ids)
            assert len(retrieved) == 1
            assert retrieved[0]["id"] == search_results[0]["id"]

        # Test health check
        health = vector_store.health_check()
        assert health["healthy"] is True
        assert health["document_count"] == len(openapi_chunks)

    def test_metadata_preservation(self, vector_store, openapi_chunks):
        """Test that complex metadata is preserved through the pipeline."""
        vector_store.setup()

        # Find a chunk with complex metadata
        complex_chunk = None
        for chunk in openapi_chunks:
            if (
                chunk["metadata"].get("ref_ids")
                and isinstance(chunk["metadata"]["ref_ids"], dict)
                and chunk["metadata"]["ref_ids"]
            ):
                complex_chunk = chunk
                break

        if not complex_chunk:
            pytest.skip("No chunks with complex ref_ids found")

        # Index the chunk
        vector_store.add_chunks([complex_chunk])

        # Retrieve it back
        retrieved = vector_store.get_by_ids([complex_chunk["id"]])
        assert len(retrieved) == 1

        original_metadata = complex_chunk["metadata"]
        retrieved_metadata = retrieved[0]["metadata"]

        # Verify metadata preservation
        assert retrieved_metadata["source_file"] == original_metadata["source_file"]
        assert retrieved_metadata["type"] == original_metadata["type"]

        # Verify complex fields are preserved
        if "ref_ids" in original_metadata:
            assert "ref_ids" in retrieved_metadata
            assert isinstance(retrieved_metadata["ref_ids"], dict)
            # Note: JSON roundtrip may not preserve exact order but content should match

        if "referenced_by" in original_metadata:
            assert "referenced_by" in retrieved_metadata
            assert isinstance(retrieved_metadata["referenced_by"], list)

    def test_search_relevance(self, vector_store, openapi_chunks):
        """Test that search returns semantically relevant results."""
        vector_store.setup()
        vector_store.add_chunks(openapi_chunks)

        # Test specific queries with expected semantic matches
        test_cases = [
            {
                "query": "user authentication login",
                "expected_terms": ["user", "auth", "login", "session", "token"],
            },
            {
                "query": "error handling HTTP status codes",
                "expected_terms": ["error", "exception", "status", "400", "500"],
            },
            {
                "query": "schema data model definition",
                "expected_terms": ["schema", "model", "definition", "properties", "component"],
            },
        ]

        for test_case in test_cases:
            results = vector_store.search(test_case["query"], limit=3)

            if results:  # Only test if we get results
                # Check that at least one result contains expected terms
                found_relevant = False
                for result in results:
                    document_lower = result["document"].lower()
                    id_lower = result["id"].lower()

                    # Check if any expected terms appear in document or ID
                    for term in test_case["expected_terms"]:
                        if term.lower() in document_lower or term.lower() in id_lower:
                            found_relevant = True
                            break

                    if found_relevant:
                        break

                # At least one result should be semantically relevant
                assert found_relevant, f"No relevant results found for query: {test_case['query']}"

    def test_batch_processing_large_dataset(self, vector_store):
        """Test batch processing with a larger subset of chunks."""
        vector_store.setup()

        # Get larger subset of chunks
        processor = OpenAPIProcessor()
        all_chunks = processor.process_directory("samples")
        large_subset = all_chunks[:200]  # Use 200 chunks

        # Process in small batches
        vector_store.add_chunks(large_subset, batch_size=50)

        # Verify all chunks were added
        info = vector_store.get_info()
        assert info["count"] == len(large_subset)

        # Test that search works across all chunks
        results = vector_store.search("API specification", limit=10)
        assert len(results) <= 10

        # Test retrieval of specific chunks from different batches
        test_ids = [large_subset[0]["id"], large_subset[100]["id"], large_subset[-1]["id"]]
        retrieved = vector_store.get_by_ids(test_ids)
        assert len(retrieved) == len(test_ids)

    def test_persistence_across_restarts(self, vector_store, openapi_chunks, temp_db_dir):
        """Test that data persists across VectorStore restarts."""
        # Initial setup and indexing
        vector_store.setup()
        vector_store.add_chunks(openapi_chunks[:50])  # Use smaller subset

        initial_count = vector_store.get_info()["count"]
        assert initial_count == 50

        # Create new VectorStoreManager instance with same configuration
        vector_store2 = VectorStoreManager(
            persist_directory=str(Path(temp_db_dir) / "chromadb"),
            collection_name="integration_test",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="cpu",
            max_tokens=256,
            reset_on_start=False,  # Don't reset - should preserve data
        )

        # Setup new instance
        vector_store2.setup()

        # Verify data persisted
        persistent_count = vector_store2.get_info()["count"]
        assert persistent_count == initial_count

        # Verify search still works
        results = vector_store2.search("API", limit=3)
        assert len(results) > 0

    def test_error_handling(self, vector_store):
        """Test error handling in various scenarios."""
        # Test operations before setup
        with pytest.raises(RuntimeError, match="not set up"):
            vector_store.add_chunks([])

        with pytest.raises(RuntimeError, match="not set up"):
            vector_store.search("test")

        with pytest.raises(RuntimeError, match="not set up"):
            vector_store.get_by_ids(["test"])

        # Setup for further tests
        vector_store.setup()

        # Test empty search
        results = vector_store.search("")
        assert results == []

        # Test non-existent ID retrieval
        results = vector_store.get_by_ids(["non_existent_id"])
        assert results == []

        # Test empty chunk list
        vector_store.add_chunks([])  # Should not raise error

        info = vector_store.get_info()
        assert info["count"] == 0
