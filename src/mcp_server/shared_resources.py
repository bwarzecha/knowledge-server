"""Shared resources for MCP server - manages pre-built indices."""

import json
from typing import Any, Dict, Optional

from src.cli.config import Config
from src.retriever import KnowledgeRetriever
from src.vector_store.vector_store_manager import VectorStoreManager


class SharedResources:
    """Manages shared resources loaded from pre-built indices."""

    def __init__(self):
        self.vector_store: Optional[VectorStoreManager] = None
        self.retriever: Optional[KnowledgeRetriever] = None
        self.api_index: Optional[Dict[str, Any]] = None
        self.config: Optional[Config] = None

    def load_from_config(self, config: Config):
        """Load all shared resources from configuration."""
        self.config = config

        # Load vector store
        self.vector_store = VectorStoreManager(
            persist_directory=config.vector_store_dir,
            collection_name=config.vector_store_collection,
            embedding_model_name=config.embedding_model,
            embedding_device=config.embedding_device,
            max_tokens=config.max_tokens,
            reset_on_start=False,  # Use existing data
        )
        self.vector_store.setup()

        # Load API index
        with open(config.api_index_path, "r") as f:
            self.api_index = json.load(f)

        # Initialize retriever with environment-based config
        self.retriever = KnowledgeRetriever(self.vector_store)

    def is_ready(self) -> bool:
        """Check if all resources are loaded and ready."""
        return self.vector_store is not None and self.retriever is not None and self.api_index is not None


# Global shared resources instance
_shared_resources = SharedResources()


def get_shared_resources() -> SharedResources:
    """Get the global shared resources instance."""
    return _shared_resources


def initialize_shared_resources(config: Config):
    """Initialize shared resources from configuration."""
    _shared_resources.load_from_config(config)
