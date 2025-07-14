"""Utilities for embedding generation with stateless functions."""

import logging
from typing import List, Optional

import tiktoken
from sentence_transformers import SentenceTransformer

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str, device: str = "mps") -> SentenceTransformer:
    """
    Load embedding model with specified device.

    Args:
        model_name: Name of the embedding model (e.g., "dunzhang/stella_en_1.5B_v5")
        device: Device to use ("mps", "cuda", or "cpu")

    Returns:
        Loaded SentenceTransformer model

    Raises:
        Exception: If model fails to load
    """
    try:
        logger.info(f"Loading embedding model {model_name} on device {device}")
        # For Stella models, need trust_remote_code=True
        if "stella" in model_name.lower():
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name, device=device)
        logger.info(f"Successfully loaded model {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Get accurate token count for text using tiktoken.

    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4)

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to get token count with tiktoken: {e}. Using character approximation.")
        # Fallback to character approximation
        return len(text) // 4


def trim_text_to_token_limit(text: str, max_tokens: int = 512, encoding_name: str = "cl100k_base") -> str:
    """
    Trim text to exact token limit using tiktoken.

    Args:
        text: Text to trim
        max_tokens: Maximum number of tokens (default: 512 for small models)
        encoding_name: Tiktoken encoding to use (default: cl100k_base)

    Returns:
        Trimmed text that fits within token limit
    """
    if max_tokens <= 0:
        return ""

    # Get current token count
    current_tokens = get_token_count(text, encoding_name)

    if current_tokens <= max_tokens:
        return text

    # Need to trim - use binary search for efficiency
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Trim tokens and decode back to text
        # Reserve 1 token for ellipsis
        trimmed_tokens = tokens[: max_tokens - 1]
        trimmed_text = encoding.decode(trimmed_tokens) + "..."

        logger.warning(
            f"Text trimmed from {current_tokens} to "
            f"{get_token_count(trimmed_text, encoding_name)} tokens (limit: {max_tokens})"
        )
        return trimmed_text

    except Exception as e:
        logger.warning(f"Failed to trim with tiktoken: {e}. Using character approximation.")
        # Fallback to character approximation
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        trimmed = text[: max_chars - 3] + "..."
        logger.warning(f"Text trimmed from {len(text)} to {len(trimmed)} characters (token limit: {max_tokens})")
        return trimmed


def encode_documents(
    texts: List[str],
    model: SentenceTransformer,
    max_tokens: Optional[int] = None,
    encoding_name: str = "cl100k_base",
    cache: Optional[EmbeddingCache] = None,
    content_hashes: Optional[List[str]] = None,
) -> List[List[float]]:
    """
    Encode documents for storage in vector database.

    Args:
        texts: List of document texts to encode
        model: Loaded SentenceTransformer model
        max_tokens: Optional token limit per document (will trim if specified)
        encoding_name: Tiktoken encoding to use for token counting
        cache: Optional embedding cache to use
        content_hashes: Optional list of content hashes for caching

    Returns:
        List of embedding vectors (one per document)
    """
    if not texts:
        return []

    # Trim texts if token limit specified
    processed_texts = texts
    if max_tokens:
        processed_texts = [trim_text_to_token_limit(text, max_tokens, encoding_name) for text in texts]

    # If cache is provided, try to get cached embeddings
    if cache and content_hashes and len(content_hashes) == len(texts):
        # Try to get model name from various attributes
        model_name = str(model)
        if hasattr(model, "model_card_data") and isinstance(model.model_card_data, dict):
            model_name = model.model_card_data.get("model_name", model_name)
        elif hasattr(model, "model_name"):
            model_name = model.model_name
        cached_embeddings, miss_indices = cache.get_embeddings_batch(model_name, content_hashes)

        # If all embeddings are cached, return them
        if not miss_indices:
            logger.info(f"Cache hit for all {len(texts)} embeddings")
            return [emb.tolist() for emb in cached_embeddings]

        # Compute only missing embeddings
        if miss_indices:
            logger.info(f"Cache hit for {len(texts) - len(miss_indices)}/{len(texts)} embeddings")
            texts_to_encode = [processed_texts[i] for i in miss_indices]

            try:
                # Generate embeddings for missing documents
                new_embeddings = model.encode(texts_to_encode)

                # Store new embeddings in cache
                hashes_to_cache = [content_hashes[i] for i in miss_indices]
                cache.set_embeddings_batch(model_name, hashes_to_cache, [emb for emb in new_embeddings])

                # Combine cached and new embeddings
                result = []
                new_emb_idx = 0
                for i in range(len(texts)):
                    if cached_embeddings[i] is not None:
                        result.append(cached_embeddings[i].tolist())
                    else:
                        result.append(new_embeddings[new_emb_idx].tolist())
                        new_emb_idx += 1

                return result
            except Exception as e:
                logger.error(f"Failed to encode {len(texts_to_encode)} documents: {e}")
                raise

    # No cache or cache miss for all - compute all embeddings
    try:
        # Generate embeddings for documents
        # For Stella models, documents don't need prompts (they're the "docs")
        embeddings = model.encode(processed_texts)

        # Store in cache if provided
        if cache and content_hashes and len(content_hashes) == len(texts):
            # Try to get model name from various attributes
            model_name = str(model)
            if hasattr(model, "model_card_data") and isinstance(model.model_card_data, dict):
                model_name = model.model_card_data.get("model_name", model_name)
            elif hasattr(model, "model_name"):
                model_name = model.model_name
            cache.set_embeddings_batch(model_name, content_hashes, [emb for emb in embeddings])

        # Convert numpy array to list of lists - SentenceTransformer returns numpy array by default
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Failed to encode {len(texts)} documents: {e}")
        raise


def encode_query(
    query: str,
    model: SentenceTransformer,
    max_tokens: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> List[float]:
    """
    Encode query for semantic search.

    For Stella models, uses "s2p_query" prompt for sentence-to-passage search.
    For Qwen3 models, uses "query" prompt for optimization.

    Args:
        query: Search query text
        model: Loaded SentenceTransformer model
        max_tokens: Optional token limit for query (will trim if specified)
        encoding_name: Tiktoken encoding to use for token counting

    Returns:
        Query embedding vector
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Trim query if token limit specified
    processed_query = query
    if max_tokens:
        processed_query = trim_text_to_token_limit(query, max_tokens, encoding_name)

    try:
        model_name = str(model).lower()

        # For Arctic-Embed models, prepend query prefix for better retrieval
        if "arctic-embed" in model_name:
            query_prefix = "Represent this sentence for searching relevant passages: "
            processed_query = query_prefix + processed_query
            logger.debug("Using Arctic-Embed query prefix")

        # For Stella models, use s2p_query prompt for sentence-to-passage search
        elif "stella" in model_name:
            try:
                embedding = model.encode([processed_query], prompt_name="s2p_query")[0]
                return embedding.tolist()
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to use s2p_query prompt for Stella model: {e}. Using default encoding.")

        # For Qwen3 models, use query prompt for better search performance
        elif "qwen" in model_name:
            try:
                embedding = model.encode([processed_query], prompt_name="query")[0]
                return embedding.tolist()
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to use query prompt for Qwen model: {e}. Using default encoding.")

        # Standard encoding without prompt (fallback for all models)
        embedding = model.encode([processed_query])[0]
        return embedding.tolist()

    except Exception as e:
        logger.error(f"Failed to encode query '{query}': {e}")
        raise


def validate_embedding_dimensions(embeddings: List[List[float]], expected_dim: Optional[int] = None) -> bool:
    """
    Validate that embeddings have consistent dimensions.

    Args:
        embeddings: List of embedding vectors
        expected_dim: Expected dimension (if None, uses first embedding's dimension)

    Returns:
        True if all embeddings have same dimension, False otherwise
    """
    if not embeddings:
        return True

    # Get expected dimension from first embedding if not provided
    if expected_dim is None:
        expected_dim = len(embeddings[0])

    # Check all embeddings have expected dimension
    for i, embedding in enumerate(embeddings):
        if len(embedding) != expected_dim:
            logger.error(f"Embedding {i} has dimension {len(embedding)}, expected {expected_dim}")
            return False

    return True
