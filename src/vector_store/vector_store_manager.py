"""Main Vector Store Manager for orchestrating all vector database operations."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chromadb_utils import (
    SentenceTransformerEmbeddingFunction,
    add_chunks_to_collection,
    create_chromadb_client,
    create_collection,
    get_chunks_by_ids,
    get_collection_info,
    search_collection,
)
from .embedding_utils import load_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Main orchestrator for vector store operations."""

    def __init__(
        self,
        persist_directory: str = "./chromadb_data",
        collection_name: str = "api_knowledge",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_device: str = "mps",
        max_tokens: int = 32000,
        reset_on_start: bool = False,
    ):
        """
        Initialize Vector Store Manager.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_model_name: Name of the embedding model to use
            embedding_device: Device for embedding model (mps, cuda, cpu)
            max_tokens: Maximum tokens per document for embeddings
            reset_on_start: Whether to reset collection on startup
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.max_tokens = max_tokens
        self.reset_on_start = reset_on_start

        # Initialize components
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.embedding_function = None

        logger.info(f"Initialized VectorStoreManager with model {embedding_model_name}")

    @classmethod
    def from_env(cls) -> "VectorStoreManager":
        """
        Create VectorStoreManager from environment variables.

        Returns:
            Configured VectorStoreManager instance
        """
        return cls(
            persist_directory=os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_data"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "api_knowledge"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "mps"),
            max_tokens=int(os.getenv("MAX_EMBEDDING_TOKENS", "256")),
            reset_on_start=os.getenv("CHROMA_RESET_ON_START", "false").lower() == "true",
        )

    def setup(self) -> None:
        """Set up all components (client, model, collection)."""
        logger.info("Setting up VectorStoreManager")

        # Create persistence directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = create_chromadb_client(self.persist_directory)

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = load_embedding_model(self.embedding_model_name, device=self.embedding_device)

        # Create embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(self.embedding_model, max_tokens=self.max_tokens)

        # Create or get collection
        self.collection = create_collection(
            self.client, self.collection_name, self.embedding_function, reset=self.reset_on_start
        )

        logger.info("VectorStoreManager setup complete")

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks with id, document, and metadata
            batch_size: Number of chunks to process in each batch
        """
        if not self.collection:
            raise RuntimeError("VectorStoreManager not set up. Call setup() first.")

        logger.info(f"Adding {len(chunks)} chunks to vector store")
        add_chunks_to_collection(self.collection, chunks, batch_size)

        # Log final count
        info = self.get_info()
        logger.info(f"Vector store now contains {info['count']} documents")

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store using semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with id, document, metadata, distance, and rank
        """
        if not self.collection or not self.embedding_model:
            raise RuntimeError("VectorStoreManager not set up. Call setup() first.")

        return search_collection(
            self.collection,
            query,
            self.embedding_model,
            limit=limit,
            filters=filters,
            max_tokens=self.max_tokens,
        )

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs to retrieve

        Returns:
            List of chunks with id, document, and metadata
        """
        if not self.collection:
            raise RuntimeError("VectorStoreManager not set up. Call setup() first.")

        return get_chunks_by_ids(self.collection, ids)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            raise RuntimeError("VectorStoreManager not set up. Call setup() first.")

        info = get_collection_info(self.collection)
        info.update(
            {
                "embedding_model": self.embedding_model_name,
                "embedding_device": self.embedding_device,
                "max_tokens": self.max_tokens,
                "persist_directory": self.persist_directory,
            }
        )

        return info

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if not self.client:
            raise RuntimeError("VectorStoreManager not set up. Call setup() first.")

        logger.warning(f"Clearing collection {self.collection_name}")

        # Delete and recreate collection
        self.collection = create_collection(self.client, self.collection_name, self.embedding_function, reset=True)

        logger.info("Collection cleared")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.

        Returns:
            Health status information
        """
        status = {"client": False, "embedding_model": False, "collection": False, "errors": []}

        try:
            # Check client
            if self.client:
                status["client"] = True
            else:
                status["errors"].append("ChromaDB client not initialized")

            # Check embedding model
            if self.embedding_model:
                # Test encoding
                test_embedding = self.embedding_model.encode(["test"])
                if len(test_embedding) > 0:
                    status["embedding_model"] = True
                else:
                    status["errors"].append("Embedding model produces empty results")
            else:
                status["errors"].append("Embedding model not loaded")

            # Check collection
            if self.collection:
                info = get_collection_info(self.collection)
                status["collection"] = True
                status["document_count"] = info["count"]
            else:
                status["errors"].append("ChromaDB collection not created")

        except Exception as e:
            status["errors"].append(f"Health check error: {str(e)}")

        status["healthy"] = len(status["errors"]) == 0

        return status
