"""Utilities for ChromaDB collection management and operations."""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from .embedding_utils import encode_documents, encode_query
from .metadata_utils import prepare_metadata_for_chromadb, restore_metadata_from_chromadb

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using SentenceTransformers."""

    def __init__(self, model: SentenceTransformer, max_tokens: Optional[int] = None):
        """
        Initialize with a SentenceTransformer model.

        Args:
            model: Loaded SentenceTransformer model
            max_tokens: Optional token limit for embeddings
        """
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function."""
        return "sentence_transformer_custom"

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the embedding function."""
        # Try to get the actual model name from various possible attributes
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default fallback
        if hasattr(self.model, "_modules") and "sentence_bert" in self.model._modules:
            # For newer SentenceTransformer models
            model_name = getattr(self.model, "model_name", model_name)
        elif hasattr(self.model, "name"):
            model_name = self.model.name
        elif hasattr(self.model, "config") and hasattr(self.model.config, "name_or_path"):
            model_name = self.model.config.name_or_path

        return {"model_name": model_name, "max_tokens": self.max_tokens}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "SentenceTransformerEmbeddingFunction":
        """Build embedding function from configuration."""
        model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        max_tokens = config.get("max_tokens")

        # Load the model
        model = SentenceTransformer(model_name)

        return cls(model=model, max_tokens=max_tokens)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Encode documents for ChromaDB storage.

        Args:
            input: List of document texts to encode

        Returns:
            List of embedding vectors
        """
        return encode_documents(input, self.model, max_tokens=self.max_tokens)


def create_chromadb_client(persist_directory: str = "./chromadb_data") -> chromadb.Client:
    """
    Create ChromaDB client with persistent storage.

    Args:
        persist_directory: Directory to persist ChromaDB data

    Returns:
        ChromaDB client instance
    """
    try:
        settings = Settings(
            is_persistent=True,
            persist_directory=persist_directory,
            anonymized_telemetry=False,  # Disable telemetry for privacy
        )

        client = chromadb.Client(settings)
        logger.info(f"Created ChromaDB client with persistence at {persist_directory}")
        return client

    except Exception as e:
        logger.error(f"Failed to create ChromaDB client: {e}")
        raise


def create_collection(
    client: chromadb.Client,
    collection_name: str,
    embedding_function: EmbeddingFunction,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Create or get ChromaDB collection.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        embedding_function: Function to generate embeddings
        reset: Whether to delete and recreate existing collection

    Returns:
        ChromaDB collection instance
    """
    try:
        # Check if collection exists
        existing_collections = client.list_collections()
        collection_exists = any(col.name == collection_name for col in existing_collections)

        if collection_exists and reset:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
            collection_exists = False

        if collection_exists:
            logger.info(f"Using existing collection: {collection_name}")
            collection = client.get_collection(
                name=collection_name, embedding_function=embedding_function
            )
        else:
            logger.info(f"Creating new collection: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

        return collection

    except Exception as e:
        logger.error(f"Failed to create/get collection {collection_name}: {e}")
        raise


def add_chunks_to_collection(
    collection: chromadb.Collection, chunks: List[Dict[str, Any]], batch_size: int = 100
) -> None:
    """
    Add chunks to ChromaDB collection in batches.

    Args:
        collection: ChromaDB collection
        chunks: List of chunks with id, document, and metadata
        batch_size: Number of chunks to process in each batch
    """
    if not chunks:
        logger.info("No chunks to add")
        return

    logger.info(f"Adding {len(chunks)} chunks to collection in batches of {batch_size}")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        try:
            # Prepare batch data
            ids = [chunk["id"] for chunk in batch]
            documents = [chunk["document"] for chunk in batch]
            metadatas = [prepare_metadata_for_chromadb(chunk["metadata"]) for chunk in batch]

            # Add to collection
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            logger.info(
                f"Added batch {batch_num}/{total_batches}: chunks {i+1}-{min(i+batch_size, len(chunks))}"
            )

        except Exception as e:
            logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
            raise


def search_collection(
    collection: chromadb.Collection,
    query: str,
    embedding_model: SentenceTransformer,
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search collection using semantic similarity.

    Args:
        collection: ChromaDB collection to search
        query: Search query text
        embedding_model: Model to encode the query
        limit: Maximum number of results to return
        filters: Optional metadata filters
        max_tokens: Optional token limit for query

    Returns:
        List of search results with id, document, metadata, distance, and rank
    """
    if not query.strip():
        return []

    try:
        # Encode query
        query_embedding = encode_query(query, embedding_model, max_tokens=max_tokens)

        # Build where clause for filters
        where_clause = None
        if filters:
            where_clause = _build_where_clause(filters)

        # Perform search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:  # Check if we have results
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": restore_metadata_from_chromadb(results["metadatas"][0][i]),
                    "distance": results["distances"][0][i],
                    "rank": i + 1,
                }
                formatted_results.append(result)

        logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
        return formatted_results

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise


def get_chunks_by_ids(collection: chromadb.Collection, ids: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve chunks by their IDs.

    Args:
        collection: ChromaDB collection
        ids: List of chunk IDs to retrieve

    Returns:
        List of chunks with id, document, and metadata
    """
    if not ids:
        return []

    try:
        results = collection.get(ids=ids, include=["documents", "metadatas"])

        # Format results
        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunk = {
                "id": chunk_id,
                "document": results["documents"][i],
                "metadata": restore_metadata_from_chromadb(results["metadatas"][i]),
            }
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks by ID")
        return chunks

    except Exception as e:
        logger.error(f"Failed to retrieve chunks by IDs {ids}: {e}")
        raise


def get_collection_info(collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Get information about a collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Dictionary with collection statistics
    """
    try:
        count = collection.count()

        info = {"name": collection.name, "count": count, "metadata": collection.metadata}

        logger.info(f"Collection {collection.name} has {count} documents")
        return info

    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise


def _build_where_clause(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ChromaDB where clause from filters.

    Args:
        filters: Dictionary of filter conditions

    Returns:
        ChromaDB-compatible where clause
    """
    where_clause = {}

    for key, value in filters.items():
        # Simple equality filters for now
        where_clause[key] = {"$eq": value}

    return where_clause
