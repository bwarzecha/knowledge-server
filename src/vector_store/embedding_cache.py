"""Simple embedding cache using SQLite."""

import hashlib
import pickle
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np


class EmbeddingCache:
    """Simple SQLite-based cache for embeddings."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize cache with optional custom path."""
        if cache_path is None:
            cache_path = Path("data/embedding_cache.db")

        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """
            )
            conn.commit()

    def _compute_cache_key(self, model_name: str, content_hash: str) -> str:
        """Compute cache key from model name and content hash."""
        combined = f"{model_name}:{content_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_embedding(self, model_name: str, content_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding if it exists."""
        cache_key = self._compute_cache_key(model_name, content_hash)

        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE cache_key = ?", (cache_key,)
            )
            row = cursor.fetchone()

            if row:
                return pickle.loads(row[0])
            return None

    def set_embedding(self, model_name: str, content_hash: str, embedding: np.ndarray):
        """Store embedding in cache."""
        cache_key = self._compute_cache_key(model_name, content_hash)
        embedding_bytes = pickle.dumps(embedding)

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, embedding) VALUES (?, ?)",
                (cache_key, embedding_bytes),
            )
            conn.commit()

    def get_embeddings_batch(
        self, model_name: str, content_hashes: List[str]
    ) -> tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get multiple embeddings from cache.

        Returns:
            - List of embeddings (None for cache misses)
            - List of indices where cache misses occurred
        """
        embeddings = []
        miss_indices = []

        for i, content_hash in enumerate(content_hashes):
            embedding = self.get_embedding(model_name, content_hash)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def set_embeddings_batch(
        self, model_name: str, content_hashes: List[str], embeddings: List[np.ndarray]
    ):
        """Store multiple embeddings in cache."""
        for content_hash, embedding in zip(content_hashes, embeddings):
            self.set_embedding(model_name, content_hash, embedding)
