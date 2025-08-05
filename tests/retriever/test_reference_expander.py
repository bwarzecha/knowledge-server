"""Tests for ReferenceExpander using real chunk data."""

import pytest

from src.retriever.reference_expander import ReferenceExpander
from tests.retriever.test_helpers import (MockVectorStore,
                                          get_chunks_with_references,
                                          get_circular_reference_chunks,
                                          get_real_test_chunks)


class TestReferenceExpander:
    """Test ReferenceExpander with real OpenAPI chunks."""

    @pytest.fixture
    def real_chunks(self):
        """Get real chunks for testing."""
        return get_real_test_chunks()

    @pytest.fixture
    def mock_vector_store(self, real_chunks):
        """Create mock vector store with real chunks."""
        return MockVectorStore(real_chunks)

    @pytest.fixture
    def reference_expander(self, mock_vector_store):
        """Create ReferenceExpander with mock vector store."""
        return ReferenceExpander(mock_vector_store)

    def test_expand_references_basic(self, reference_expander):
        """Test basic reference expansion with real chunks."""
        # Get a chunk that has references
        chunks_with_refs = get_chunks_with_references()
        assert len(chunks_with_refs) > 0, "Need chunks with references for this test"

        primary_chunk = chunks_with_refs[0]
        ref_ids = primary_chunk["metadata"]["ref_ids"]

        print(f"Testing with chunk: {primary_chunk['id']}")
        print(f"References: {list(ref_ids.keys())}")

        # Expand references
        referenced_chunks, stats = reference_expander.expand_references(
            primary_chunks=[primary_chunk], max_depth=2, max_total=10
        )

        # Verify results
        assert isinstance(referenced_chunks, list)
        assert isinstance(stats.expansion_time_ms, float)
        assert stats.expansion_time_ms >= 0
        assert stats.primary_count == 1
        assert stats.referenced_count == len(referenced_chunks)

        # Should find at least some of the referenced chunks
        if ref_ids:
            assert (
                len(referenced_chunks) > 0
            ), "Should find at least one referenced chunk"

        print(f"Found {len(referenced_chunks)} referenced chunks")
        for chunk in referenced_chunks:
            print(f"  - {chunk['id']}")

    def test_circular_reference_protection(self, reference_expander):
        """Test that circular references are detected and handled."""
        try:
            # Get chunks that reference each other
            operation_chunk, error_chunk = get_circular_reference_chunks()

            print(
                f"Testing circular refs: {operation_chunk['id']} <-> {error_chunk['id']}"
            )

            # Start expansion from operation chunk
            referenced_chunks, stats = reference_expander.expand_references(
                primary_chunks=[operation_chunk], max_depth=3, max_total=10
            )

            # Should detect circular references
            assert stats.circular_refs_detected > 0, "Should detect circular references"
            print(f"Detected {stats.circular_refs_detected} circular references")

            # Should still return valid results
            assert isinstance(referenced_chunks, list)
            assert stats.referenced_count == len(referenced_chunks)

        except ValueError as e:
            if "Could not find circular reference chunks" in str(e):
                pytest.skip("No circular reference chunks available in test data")
            else:
                raise

    def test_depth_limiting(self, reference_expander):
        """Test that max_depth parameter limits expansion depth."""
        chunks_with_refs = get_chunks_with_references()
        if not chunks_with_refs:
            pytest.skip("No chunks with references available")

        primary_chunk = chunks_with_refs[0]

        # Test with depth limit of 1
        referenced_chunks_d1, stats_d1 = reference_expander.expand_references(
            primary_chunks=[primary_chunk], max_depth=1, max_total=20
        )

        # Test with depth limit of 2
        referenced_chunks_d2, stats_d2 = reference_expander.expand_references(
            primary_chunks=[primary_chunk], max_depth=2, max_total=20
        )

        # Verify depth limits
        assert stats_d1.depth_reached <= 1
        assert stats_d2.depth_reached <= 2

        # Usually deeper search should find same or more chunks
        assert len(referenced_chunks_d2) >= len(referenced_chunks_d1)

        print(
            f"Depth 1: {len(referenced_chunks_d1)} chunks, max depth {stats_d1.depth_reached}"
        )
        print(
            f"Depth 2: {len(referenced_chunks_d2)} chunks, max depth {stats_d2.depth_reached}"
        )

    def test_count_limiting(self, reference_expander):
        """Test that max_total parameter limits number of results."""
        chunks_with_refs = get_chunks_with_references()
        if not chunks_with_refs:
            pytest.skip("No chunks with references available")

        # Use multiple primary chunks to increase reference potential
        primary_chunks = chunks_with_refs[:3]

        # Test with small limit
        referenced_chunks_small, stats_small = reference_expander.expand_references(
            primary_chunks=primary_chunks, max_depth=3, max_total=2
        )

        # Test with larger limit
        referenced_chunks_large, stats_large = reference_expander.expand_references(
            primary_chunks=primary_chunks, max_depth=3, max_total=10
        )

        # Verify count limits
        assert len(referenced_chunks_small) <= 2
        assert len(referenced_chunks_large) <= 10
        assert len(referenced_chunks_large) >= len(referenced_chunks_small)

        print(f"Small limit (2): {len(referenced_chunks_small)} chunks")
        print(f"Large limit (10): {len(referenced_chunks_large)} chunks")

    def test_empty_primary_chunks(self, reference_expander):
        """Test expansion with empty primary chunks list."""
        referenced_chunks, stats = reference_expander.expand_references(
            primary_chunks=[], max_depth=3, max_total=10
        )

        assert referenced_chunks == []
        assert stats.primary_count == 0
        assert stats.referenced_count == 0
        assert stats.depth_reached == 0
        assert stats.circular_refs_detected == 0

    def test_chunks_without_references(self, reference_expander, real_chunks):
        """Test expansion with chunks that have no references."""
        # Find chunks without references
        chunks_without_refs = [
            chunk for chunk in real_chunks if not chunk["metadata"].get("ref_ids")
        ]

        if not chunks_without_refs:
            pytest.skip("All chunks have references")

        primary_chunk = chunks_without_refs[0]

        referenced_chunks, stats = reference_expander.expand_references(
            primary_chunks=[primary_chunk], max_depth=3, max_total=10
        )

        assert referenced_chunks == []
        assert stats.primary_count == 1
        assert stats.referenced_count == 0
        assert stats.depth_reached == 0

    def test_missing_referenced_chunks(self, real_chunks):
        """Test handling of missing referenced chunks."""
        # Create chunk with reference to non-existent chunk
        fake_chunk = {
            "id": "fake:chunk",
            "document": "Fake chunk for testing",
            "metadata": {
                "type": "test",
                "ref_ids": {"non-existent:chunk": [], "another-missing:chunk": []},
            },
        }

        # Create vector store without the referenced chunks
        mock_vector_store = MockVectorStore(real_chunks)
        reference_expander = ReferenceExpander(mock_vector_store)

        referenced_chunks, stats = reference_expander.expand_references(
            primary_chunks=[fake_chunk], max_depth=2, max_total=10
        )

        # Should handle missing chunks gracefully
        assert isinstance(referenced_chunks, list)
        assert stats.primary_count == 1
        # Should not find any referenced chunks since they don't exist
        assert len(referenced_chunks) == 0
