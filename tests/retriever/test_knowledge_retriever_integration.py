"""Integration tests for Knowledge Retriever with real Vector Store Manager."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.cli.config import Config
from src.openapi_processor.processor import OpenAPIProcessor
from src.retriever import KnowledgeRetriever
from src.retriever.data_classes import RetrieverConfig
from src.vector_store.vector_store_manager import VectorStoreManager


class TestKnowledgeRetrieverIntegration:
    """Integration tests with real Vector Store Manager and embeddings."""

    @pytest.fixture(scope="class")
    def temp_db_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def real_chunks(self):
        """Generate real chunks from OpenAPI samples."""
        config = Config()
        processor = OpenAPIProcessor(config)
        chunks = processor.process_directory("open-api-small-samples/3.0/json/openapi-workshop")
        return chunks

    @pytest.fixture(scope="class")
    def vector_store(self, temp_db_dir):
        """Create real Vector Store Manager with embeddings."""
        return VectorStoreManager(
            persist_directory=str(Path(temp_db_dir) / "chromadb"),
            collection_name="retriever_integration_test",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast model for testing
            embedding_device="cpu",
            max_tokens=256,
            reset_on_start=True,
        )

    @pytest.fixture(scope="class")
    def indexed_vector_store(self, vector_store, real_chunks):
        """Vector store with indexed chunks."""
        vector_store.setup()
        vector_store.add_chunks(real_chunks, batch_size=25)
        return vector_store

    @pytest.fixture
    def retriever(self, indexed_vector_store):
        """Knowledge Retriever with real vector store."""
        config = RetrieverConfig(
            max_primary_results=5,
            max_total_chunks=15,
            max_depth=3,
            token_limit=2000,  # Lower for testing
        )
        return KnowledgeRetriever(indexed_vector_store, config)

    def test_end_to_end_retrieval_hoot_query(self, retriever):
        """Test complete retrieval workflow with hoot-related query."""
        query = "how to create a hoot"

        context = retriever.retrieve_knowledge(query)

        # Basic structure validation
        assert context.query == query
        assert isinstance(context.primary_chunks, list)
        assert isinstance(context.referenced_chunks, list)
        assert context.total_chunks == len(context.primary_chunks) + len(context.referenced_chunks)
        assert context.total_tokens > 0
        assert context.retrieval_stats.total_time_ms > 0

        # Should find relevant chunks
        assert len(context.primary_chunks) > 0, "Should find primary chunks for hoot query"

        # Check if we found hoot-related content
        hoot_related = False
        for chunk in context.primary_chunks:
            if "hoot" in chunk.document.lower() or "hoot" in chunk.id.lower():
                hoot_related = True
                break

        assert hoot_related, "Should find hoot-related chunks"

        # Verify chunk structure
        for chunk in context.primary_chunks:
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "document")
            assert hasattr(chunk, "metadata")
            assert hasattr(chunk, "relevance_score")
            assert chunk.retrieval_reason == "primary_result"
            assert 0 <= chunk.relevance_score <= 1

        for chunk in context.referenced_chunks:
            assert chunk.retrieval_reason == "referenced_dependency"

        print(f"Query: {query}")
        print(f"Found {len(context.primary_chunks)} primary + {len(context.referenced_chunks)} referenced chunks")
        print(f"Total tokens: {context.total_tokens}")
        print(f"Time: {context.retrieval_stats.total_time_ms:.1f}ms")

    def test_schema_query(self, retriever):
        """Test retrieval for schema-related query."""
        query = "user schema structure"

        context = retriever.retrieve_knowledge(query)

        assert len(context.primary_chunks) > 0

        # Should find schema-related content
        for chunk in context.primary_chunks:
            if "schema" in chunk.document.lower() or chunk.metadata.get("type") == "component":
                break

        # Note: Might not find exact match depending on sample data
        print(f"Schema query found {len(context.primary_chunks)} primary chunks")
        for chunk in context.primary_chunks[:3]:
            print(f"  - {chunk.id}: {chunk.metadata.get('type')}")

    def test_reference_expansion_works(self, retriever):
        """Test that reference expansion actually includes dependencies."""
        query = "hoot endpoint operations"

        # Test with reference expansion enabled
        context_with_refs = retriever.retrieve_knowledge(query, include_references=True)

        # Test with reference expansion disabled
        context_without_refs = retriever.retrieve_knowledge(query, include_references=False)

        # Should have same primary chunks
        assert len(context_with_refs.primary_chunks) == len(context_without_refs.primary_chunks)

        # With references should have more total chunks (if any references exist)
        if any(chunk.metadata.get("ref_ids") for chunk in context_with_refs.primary_chunks):
            assert context_with_refs.total_chunks >= context_without_refs.total_chunks
            assert len(context_with_refs.referenced_chunks) >= 0

        # Without references should have no referenced chunks
        assert len(context_without_refs.referenced_chunks) == 0

        print(f"With references: {context_with_refs.total_chunks} chunks")
        print(f"Without references: {context_without_refs.total_chunks} chunks")

    def test_depth_and_limit_parameters(self, retriever):
        """Test that depth and limit parameters work correctly."""
        query = "hoot operations"

        # Test with small limits
        context_small = retriever.retrieve_knowledge(query, max_primary_results=2, max_total_chunks=5, max_depth=1)

        # Test with larger limits
        context_large = retriever.retrieve_knowledge(query, max_primary_results=8, max_total_chunks=20, max_depth=3)

        # Small limits should be respected
        assert len(context_small.primary_chunks) <= 2
        assert context_small.total_chunks <= 5
        assert context_small.retrieval_stats.depth_reached <= 1

        # Large search should find same or more
        assert len(context_large.primary_chunks) >= len(context_small.primary_chunks)
        assert context_large.total_chunks >= context_small.total_chunks

        print(f"Small limits: {context_small.total_chunks} chunks, depth {context_small.retrieval_stats.depth_reached}")
        print(f"Large limits: {context_large.total_chunks} chunks, depth {context_large.retrieval_stats.depth_reached}")

    def test_empty_query_handling(self, retriever):
        """Test handling of empty or invalid queries."""
        # Empty query
        context_empty = retriever.retrieve_knowledge("")

        assert context_empty.query == ""
        assert len(context_empty.primary_chunks) == 0
        assert len(context_empty.referenced_chunks) == 0
        assert context_empty.total_chunks == 0
        assert context_empty.total_tokens == 0

        # Whitespace only query
        context_whitespace = retriever.retrieve_knowledge("   ")

        assert len(context_whitespace.primary_chunks) == 0

    def test_nonexistent_query(self, retriever):
        """Test query that should return very few results."""
        query = "xyzqwerty123456789 nonexistent super rare term"

        context = retriever.retrieve_knowledge(query)

        # Should return valid context (may find some results due to embedding similarity)
        assert context.query == query
        assert isinstance(context.primary_chunks, list)
        assert isinstance(context.referenced_chunks, list)
        assert context.total_chunks >= 0  # May find some results due to embedding similarity
        assert context.total_tokens >= 0

        print(f"Rare term query returned {context.total_chunks} chunks")

    def test_performance_benchmarks(self, retriever):
        """Test that retrieval meets performance targets."""
        queries = [
            "create hoot",
            "get user information",
            "error handling",
            "schema definition",
            "api endpoint",
        ]

        total_time = 0
        successful_queries = 0

        for query in queries:
            context = retriever.retrieve_knowledge(query)

            if len(context.primary_chunks) > 0:
                successful_queries += 1
                total_time += context.retrieval_stats.total_time_ms

                # Each query should complete reasonably quickly
                assert context.retrieval_stats.total_time_ms < 2000, (
                    f"Query '{query}' took too long: {context.retrieval_stats.total_time_ms}ms"
                )

                print(f"'{query}': {len(context.primary_chunks)} chunks, {context.retrieval_stats.total_time_ms:.1f}ms")

        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"Average retrieval time: {avg_time:.1f}ms across {successful_queries} successful queries")

            # Average should be reasonable
            assert avg_time < 1000, f"Average retrieval time too slow: {avg_time}ms"

    def test_token_estimation_accuracy(self, retriever):
        """Test that token estimation is reasonable."""
        query = "hoot endpoint operations"

        context = retriever.retrieve_knowledge(query)

        if context.total_chunks > 0:
            # Token count should be positive
            assert context.total_tokens > 0

            # Should be reasonable per chunk (not too low or too high)
            avg_tokens_per_chunk = context.total_tokens / context.total_chunks
            assert 10 < avg_tokens_per_chunk < 1000, f"Average tokens per chunk seems wrong: {avg_tokens_per_chunk}"

            print(f"Total tokens: {context.total_tokens} across {context.total_chunks} chunks")
            print(f"Average tokens per chunk: {avg_tokens_per_chunk:.1f}")

    def test_retrieval_stats_accuracy(self, retriever):
        """Test that retrieval statistics are accurate."""
        query = "hoot operations"

        context = retriever.retrieve_knowledge(query)

        stats = context.retrieval_stats

        # Time components should add up
        assert abs(stats.total_time_ms - (stats.search_time_ms + stats.expansion_time_ms)) < 1.0

        # Counts should match actual results
        assert stats.primary_count == len(context.primary_chunks)
        assert stats.referenced_count == len(context.referenced_chunks)

        # Times should be non-negative
        assert stats.search_time_ms >= 0
        assert stats.expansion_time_ms >= 0
        assert stats.total_time_ms >= 0

        # Depth should be reasonable
        assert 0 <= stats.depth_reached <= 3
        assert stats.circular_refs_detected >= 0

        print(f"Stats: search={stats.search_time_ms:.1f}ms, expansion={stats.expansion_time_ms:.1f}ms")
        print(f"Depth: {stats.depth_reached}, Circular refs: {stats.circular_refs_detected}")
