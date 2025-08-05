"""Tests for embedding cache functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.vector_store.embedding_cache import EmbeddingCache
from src.vector_store.embedding_utils import load_embedding_model


class TestEmbeddingCache:
    """Test cases for EmbeddingCache."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = EmbeddingCache(cache_path)

            model_name = "test_model"
            content_hash = "abc123"
            embedding = np.array([0.1, 0.2, 0.3])

            # Test cache miss
            result = cache.get_embedding(model_name, content_hash)
            assert result is None

            # Test cache set
            cache.set_embedding(model_name, content_hash, embedding)

            # Test cache hit
            result = cache.get_embedding(model_name, content_hash)
            assert result is not None
            assert np.array_equal(result, embedding)

    def test_cache_different_models(self):
        """Test that different models have separate cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = EmbeddingCache(cache_path)

            content_hash = "same_content"
            embedding1 = np.array([0.1, 0.2, 0.3])
            embedding2 = np.array([0.4, 0.5, 0.6])

            # Set embeddings for different models
            cache.set_embedding("model1", content_hash, embedding1)
            cache.set_embedding("model2", content_hash, embedding2)

            # Verify different models return different embeddings
            result1 = cache.get_embedding("model1", content_hash)
            result2 = cache.get_embedding("model2", content_hash)

            assert np.array_equal(result1, embedding1)
            assert np.array_equal(result2, embedding2)
            assert not np.array_equal(result1, result2)

    def test_cache_batch_operations(self):
        """Test batch get/set operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = EmbeddingCache(cache_path)

            model_name = "test_model"
            content_hashes = ["hash1", "hash2", "hash3"]
            embeddings = [
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
                np.array([0.5, 0.6]),
            ]

            # Initially all should be misses
            results, miss_indices = cache.get_embeddings_batch(
                model_name, content_hashes
            )
            assert all(r is None for r in results)
            assert miss_indices == [0, 1, 2]

            # Store first two embeddings
            cache.set_embeddings_batch(model_name, content_hashes[:2], embeddings[:2])

            # Now we should have partial hits
            results, miss_indices = cache.get_embeddings_batch(
                model_name, content_hashes
            )
            assert results[0] is not None and np.array_equal(results[0], embeddings[0])
            assert results[1] is not None and np.array_equal(results[1], embeddings[1])
            assert results[2] is None
            assert miss_indices == [2]

    @pytest.mark.bedrock
    def test_cache_with_real_model(self):
        """Test cache with actual embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = EmbeddingCache(cache_path)

            # Load test model
            model = load_embedding_model(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Test text
            test_text = "This is a test sentence for embedding cache."
            content_hash = "test_hash_123"

            # First encoding (cache miss)
            embedding1 = model.encode([test_text])[0]
            cache.set_embedding(model_name, content_hash, embedding1)

            # Second encoding from cache (cache hit)
            cached_embedding = cache.get_embedding(model_name, content_hash)

            # Verify embeddings are identical
            assert cached_embedding is not None
            assert np.array_equal(embedding1, cached_embedding)

    def test_cache_persistence(self):
        """Test that cache persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"

            model_name = "test_model"
            content_hash = "persistent_hash"
            embedding = np.array([0.7, 0.8, 0.9])

            # First cache instance - store embedding
            cache1 = EmbeddingCache(cache_path)
            cache1.set_embedding(model_name, content_hash, embedding)

            # Second cache instance - retrieve embedding
            cache2 = EmbeddingCache(cache_path)
            result = cache2.get_embedding(model_name, content_hash)

            assert result is not None
            assert np.array_equal(result, embedding)
