"""Tests for Knowledge Retriever data classes."""

import os


from src.retriever.data_classes import Chunk, KnowledgeContext, RetrievalStats, RetrieverConfig


class TestDataClasses:
    """Test data class functionality."""

    def test_chunk_creation(self):
        """Test Chunk data class creation."""
        chunk = Chunk(
            id="test.json:paths/users/get",
            document="GET /users endpoint for listing users",
            metadata={"type": "operation", "source_file": "test.json"},
            relevance_score=0.85,
            retrieval_reason="primary_result",
        )

        assert chunk.id == "test.json:paths/users/get"
        assert "GET /users" in chunk.document
        assert chunk.metadata["type"] == "operation"
        assert chunk.relevance_score == 0.85
        assert chunk.retrieval_reason == "primary_result"

    def test_retrieval_stats_creation(self):
        """Test RetrievalStats data class creation."""
        stats = RetrievalStats(
            search_time_ms=50.5,
            expansion_time_ms=25.2,
            total_time_ms=75.7,
            primary_count=3,
            referenced_count=5,
            depth_reached=2,
            circular_refs_detected=1,
        )

        assert stats.search_time_ms == 50.5
        assert stats.expansion_time_ms == 25.2
        assert stats.total_time_ms == 75.7
        assert stats.primary_count == 3
        assert stats.referenced_count == 5
        assert stats.depth_reached == 2
        assert stats.circular_refs_detected == 1

    def test_knowledge_context_creation(self):
        """Test KnowledgeContext data class creation."""
        primary_chunk = Chunk(
            id="test:primary",
            document="primary content",
            metadata={},
            relevance_score=0.9,
            retrieval_reason="primary_result",
        )

        referenced_chunk = Chunk(
            id="test:referenced",
            document="referenced content",
            metadata={},
            relevance_score=0.6,
            retrieval_reason="referenced_dependency",
        )

        stats = RetrievalStats(
            search_time_ms=30,
            expansion_time_ms=20,
            total_time_ms=50,
            primary_count=1,
            referenced_count=1,
            depth_reached=1,
            circular_refs_detected=0,
        )

        context = KnowledgeContext(
            query="test query",
            primary_chunks=[primary_chunk],
            referenced_chunks=[referenced_chunk],
            total_chunks=2,
            total_tokens=150,
            retrieval_stats=stats,
        )

        assert context.query == "test query"
        assert len(context.primary_chunks) == 1
        assert len(context.referenced_chunks) == 1
        assert context.total_chunks == 2
        assert context.total_tokens == 150
        assert context.retrieval_stats.primary_count == 1

    def test_retriever_config_defaults(self):
        """Test RetrieverConfig default values."""
        config = RetrieverConfig()

        assert config.max_primary_results == 5
        assert config.max_total_chunks == 15
        assert config.max_depth == 3
        assert config.timeout_ms == 5000
        assert config.token_limit == 4000
        assert config.prioritize_primary is True

    def test_retriever_config_from_env(self):
        """Test RetrieverConfig creation from environment variables."""
        # Set environment variables
        env_vars = {
            "RETRIEVAL_MAX_PRIMARY_RESULTS": "8",
            "RETRIEVAL_MAX_TOTAL_CHUNKS": "20",
            "RETRIEVAL_MAX_DEPTH": "4",
            "RETRIEVAL_TIMEOUT_MS": "6000",
            "CONTEXT_TOKEN_LIMIT": "5000",
            "CONTEXT_PRIORITIZE_PRIMARY": "false",
        }

        # Temporarily set environment variables
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = RetrieverConfig.from_env()

            assert config.max_primary_results == 8
            assert config.max_total_chunks == 20
            assert config.max_depth == 4
            assert config.timeout_ms == 6000
            assert config.token_limit == 5000
            assert config.prioritize_primary is False

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_retriever_config_from_env_with_defaults(self):
        """Test RetrieverConfig from environment with missing variables uses defaults."""
        # Ensure no retriever env vars are set
        env_keys = [
            "RETRIEVAL_MAX_PRIMARY_RESULTS",
            "RETRIEVAL_MAX_TOTAL_CHUNKS",
            "RETRIEVAL_MAX_DEPTH",
            "RETRIEVAL_TIMEOUT_MS",
            "CONTEXT_TOKEN_LIMIT",
            "CONTEXT_PRIORITIZE_PRIMARY",
        ]

        original_env = {}
        for key in env_keys:
            original_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

        try:
            config = RetrieverConfig.from_env()

            # Should use defaults
            assert config.max_primary_results == 5
            assert config.max_total_chunks == 15
            assert config.max_depth == 3
            assert config.timeout_ms == 5000
            assert config.token_limit == 4000
            assert config.prioritize_primary is True

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is not None:
                    os.environ[key] = original_value
