"""Tests for embedding utilities with real models."""

import pytest

from src.vector_store.embedding_utils import (
    encode_documents,
    encode_query,
    get_token_count,
    load_embedding_model,
    trim_text_to_token_limit,
    validate_embedding_dimensions,
)


class TestEmbeddingUtilsReal:
    """Test embedding utility functions with real embedding model."""

    @pytest.fixture(scope="class")
    def embedding_model(self):
        """Load a small, fast embedding model for testing."""
        # Use all-MiniLM-L6-v2 - it's small (80MB), fast, and supports up to 256 tokens
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            return load_embedding_model(model_name, device="cpu")  # Use CPU for reliability
        except Exception as e:
            pytest.skip(f"Could not load embedding model {model_name}: {e}")

    def test_get_token_count(self):
        """Test token counting with tiktoken."""
        # Test simple text
        text = "This is a test document"
        count = get_token_count(text)
        assert isinstance(count, int)
        assert count > 0

        # Empty text should have 0 tokens
        assert get_token_count("") == 0

        # Longer text should have more tokens
        long_text = text * 10
        long_count = get_token_count(long_text)
        assert long_count > count

    def test_trim_text_to_token_limit(self):
        """Test text trimming based on token limits."""
        # Short text should not be trimmed
        short_text = "This is a short text"
        result = trim_text_to_token_limit(short_text, max_tokens=100)
        assert result == short_text

        # Long text should be trimmed
        long_text = "word " * 100  # Create text with predictable tokens
        result = trim_text_to_token_limit(long_text, max_tokens=50)
        trimmed_count = get_token_count(result)
        assert trimmed_count <= 50
        assert result.endswith("...")

        # Edge cases
        assert trim_text_to_token_limit("", 100) == ""
        assert trim_text_to_token_limit("test", 0) == ""

    def test_encode_documents_with_real_model(self, embedding_model):
        """Test document encoding with real model."""
        texts = [
            "This is the first document about cats",
            "This is the second document about dogs",
            "A completely different topic about programming",
        ]

        embeddings = encode_documents(texts, embedding_model)

        # Check basic properties
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

        # All embeddings should have same dimension
        assert validate_embedding_dimensions(embeddings)

        # Similar documents should have more similar embeddings than different ones
        import numpy as np

        cat_emb = np.array(embeddings[0])
        dog_emb = np.array(embeddings[1])
        prog_emb = np.array(embeddings[2])

        # Cosine similarity between cat and dog should be higher than cat and programming
        cat_dog_sim = np.dot(cat_emb, dog_emb) / (np.linalg.norm(cat_emb) * np.linalg.norm(dog_emb))
        cat_prog_sim = np.dot(cat_emb, prog_emb) / (
            np.linalg.norm(cat_emb) * np.linalg.norm(prog_emb)
        )

        assert cat_dog_sim > cat_prog_sim

    def test_encode_documents_with_token_limit(self, embedding_model):
        """Test document encoding with token limiting."""
        # Create a long document that exceeds token limit
        long_text = (
            "This is a repeated sentence about machine learning and artificial intelligence. " * 20
        )
        texts = [long_text]

        # Encode with token limit
        embeddings = encode_documents(texts, embedding_model, max_tokens=50)

        # Should still produce valid embeddings
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0

    def test_encode_query_with_real_model(self, embedding_model):
        """Test query encoding with real model."""
        query = "search for information about cats"

        embedding = encode_query(query, embedding_model)

        # Check basic properties
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_encode_query_with_token_limit(self, embedding_model):
        """Test query encoding with token limiting."""
        # Create a long query
        long_query = "search for information about " + "cats and dogs " * 50

        embedding = encode_query(long_query, embedding_model, max_tokens=50)

        # Should still produce valid embedding
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_encode_query_empty_query(self, embedding_model):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            encode_query("", embedding_model)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            encode_query("   ", embedding_model)  # Only whitespace

    def test_encode_documents_empty_list(self, embedding_model):
        """Test encoding empty document list."""
        result = encode_documents([], embedding_model)
        assert result == []

    def test_similarity_consistency(self, embedding_model):
        """Test that similar texts produce similar embeddings."""
        # Test with similar sentences
        texts = [
            "The cat is sleeping on the couch",
            "A cat is resting on the sofa",
            "The dog is running in the park",
        ]

        embeddings = encode_documents(texts, embedding_model)

        import numpy as np

        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])  # Similar to emb1
        emb3 = np.array(embeddings[2])  # Different from emb1/emb2

        # Calculate cosine similarities
        sim_1_2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_1_3 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        # Similar sentences should be more similar than different ones
        assert sim_1_2 > sim_1_3
