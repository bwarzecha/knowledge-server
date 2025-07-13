"""Tests for ChromaDB utilities."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.vector_store.chromadb_utils import (
    SentenceTransformerEmbeddingFunction,
    add_chunks_to_collection,
    create_chromadb_client,
    create_collection,
    get_chunks_by_ids,
    get_collection_info,
    search_collection,
)
from src.vector_store.embedding_utils import load_embedding_model


class TestChromaDBUtils:
    """Test ChromaDB utilities with real database operations."""

    @pytest.fixture(scope="class")
    def embedding_model(self):
        """Load a small embedding model for testing."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            return load_embedding_model(model_name, device="cpu")
        except Exception as e:
            pytest.skip(f"Could not load embedding model {model_name}: {e}")

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def client(self, temp_db_dir):
        """Create ChromaDB client with temporary storage."""
        return create_chromadb_client(temp_db_dir)

    @pytest.fixture
    def embedding_function(self, embedding_model):
        """Create embedding function."""
        return SentenceTransformerEmbeddingFunction(embedding_model, max_tokens=256)

    @pytest.fixture
    def collection(self, client, embedding_function):
        """Create test collection."""
        return create_collection(client, "test_collection", embedding_function)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {
                "id": "test.json:components/schemas/User",
                "document": "User schema for user management API with properties like name, email, and ID",
                "metadata": {
                    "source_file": "test.json",
                    "element_type": "component",
                    "chunk_type": "definition",
                    "ref_ids": {"test.json:components/schemas/Address": []},
                    "referenced_by": ["test.json:paths/users/get"],
                },
            },
            {
                "id": "test.json:components/schemas/Address",
                "document": "Address schema with street, city, and postal code information",
                "metadata": {
                    "source_file": "test.json",
                    "element_type": "component",
                    "chunk_type": "definition",
                    "ref_ids": {},
                    "referenced_by": ["test.json:components/schemas/User"],
                },
            },
            {
                "id": "test.json:paths/users/get",
                "document": "GET endpoint to retrieve user information by ID",
                "metadata": {
                    "source_file": "test.json",
                    "element_type": "operation",
                    "chunk_type": "operation",
                    "ref_ids": {"test.json:components/schemas/User": []},
                    "referenced_by": [],
                },
            },
        ]

    def test_create_chromadb_client(self, temp_db_dir):
        """Test ChromaDB client creation."""
        client = create_chromadb_client(temp_db_dir)
        assert client is not None

        # Verify persistence directory was created
        assert Path(temp_db_dir).exists()

    def test_create_collection(self, client, embedding_function):
        """Test collection creation."""
        collection = create_collection(client, "test_collection", embedding_function)
        assert collection is not None
        assert collection.name == "test_collection"

        # Test getting existing collection
        collection2 = create_collection(client, "test_collection", embedding_function)
        assert collection2.name == "test_collection"

    def test_create_collection_with_reset(self, client, embedding_function):
        """Test collection creation with reset."""
        # Create initial collection
        create_collection(client, "reset_test", embedding_function)

        # Create with reset - should delete and recreate
        collection2 = create_collection(client, "reset_test", embedding_function, reset=True)
        assert collection2.name == "reset_test"

    def test_add_chunks_to_collection(self, collection, sample_chunks):
        """Test adding chunks to collection."""
        add_chunks_to_collection(collection, sample_chunks)

        # Verify chunks were added
        info = get_collection_info(collection)
        assert info["count"] == len(sample_chunks)

    def test_add_chunks_empty_list(self, collection):
        """Test adding empty list of chunks."""
        add_chunks_to_collection(collection, [])

        info = get_collection_info(collection)
        assert info["count"] == 0

    def test_get_chunks_by_ids(self, collection, sample_chunks, embedding_model):
        """Test retrieving chunks by IDs."""
        # First add chunks
        add_chunks_to_collection(collection, sample_chunks)

        # Retrieve specific chunks
        ids = ["test.json:components/schemas/User", "test.json:paths/users/get"]
        retrieved_chunks = get_chunks_by_ids(collection, ids)

        assert len(retrieved_chunks) == 2
        assert all(chunk["id"] in ids for chunk in retrieved_chunks)

        # Verify metadata was restored correctly
        user_chunk = next(c for c in retrieved_chunks if "User" in c["id"])
        assert user_chunk["metadata"]["element_type"] == "component"
        assert isinstance(user_chunk["metadata"]["ref_ids"], dict)
        assert isinstance(user_chunk["metadata"]["referenced_by"], list)

    def test_get_chunks_by_ids_empty_list(self, collection):
        """Test retrieving chunks with empty ID list."""
        chunks = get_chunks_by_ids(collection, [])
        assert chunks == []

    def test_search_collection(self, collection, sample_chunks, embedding_model):
        """Test semantic search in collection."""
        # First add chunks
        add_chunks_to_collection(collection, sample_chunks)

        # Search for user-related content
        results = search_collection(collection, "user information and profile data", embedding_model, limit=3)

        assert len(results) <= 3
        assert all("id" in result for result in results)
        assert all("document" in result for result in results)
        assert all("metadata" in result for result in results)
        assert all("distance" in result for result in results)
        assert all("rank" in result for result in results)

        # Results should be ranked by relevance
        if len(results) > 1:
            assert results[0]["rank"] == 1
            assert results[1]["rank"] == 2

    def test_search_collection_with_filters(self, collection, sample_chunks, embedding_model):
        """Test search with metadata filters."""
        # First add chunks
        add_chunks_to_collection(collection, sample_chunks)

        # Search only for components
        results = search_collection(
            collection,
            "schema definition",
            embedding_model,
            limit=5,
            filters={"element_type": "component"},
        )

        # All results should be components
        for result in results:
            assert result["metadata"]["element_type"] == "component"

    def test_search_collection_empty_query(self, collection, sample_chunks, embedding_model):
        """Test search with empty query."""
        add_chunks_to_collection(collection, sample_chunks)

        results = search_collection(collection, "", embedding_model)
        assert results == []

        results = search_collection(collection, "   ", embedding_model)
        assert results == []

    def test_get_collection_info(self, collection, sample_chunks):
        """Test getting collection information."""
        # Empty collection
        info = get_collection_info(collection)
        assert info["name"] == "test_collection"
        assert info["count"] == 0
        assert "metadata" in info

        # After adding chunks
        add_chunks_to_collection(collection, sample_chunks)
        info = get_collection_info(collection)
        assert info["count"] == len(sample_chunks)

    def test_embedding_function(self, embedding_model):
        """Test custom embedding function."""
        embedding_func = SentenceTransformerEmbeddingFunction(embedding_model, max_tokens=100)

        texts = ["test document", "another document"]
        embeddings = embedding_func(texts)

        assert len(embeddings) == len(texts)
        # ChromaDB normalizes embeddings to numpy arrays, so expect numpy arrays
        assert all(hasattr(emb, "shape") for emb in embeddings)  # numpy arrays have shape
        assert all(len(emb) > 0 for emb in embeddings)

    def test_batch_processing(self, collection, embedding_model):
        """Test batch processing with multiple batches."""
        # Create many chunks to test batching
        chunks = []
        for i in range(25):  # More than default batch size of 100, but let's use smaller batch
            chunks.append(
                {
                    "id": f"test.json:item_{i}",
                    "document": f"This is test document number {i} about various topics",
                    "metadata": {
                        "source_file": "test.json",
                        "element_type": "test",
                        "item_number": i,
                    },
                }
            )

        # Add with small batch size to test batching
        add_chunks_to_collection(collection, chunks, batch_size=10)

        # Verify all chunks were added
        info = get_collection_info(collection)
        assert info["count"] == len(chunks)

        # Test search works across all chunks
        results = search_collection(collection, "test document", embedding_model, limit=5)
        assert len(results) == 5
