"""Utilities for embedding generation with stateless functions."""

import logging
from typing import List, Optional

import tiktoken
from sentence_transformers import SentenceTransformer

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
        logger.warning(
            f"Failed to get token count with tiktoken: {e}. Using character approximation."
        )
        # Fallback to character approximation
        return len(text) // 4


def trim_text_to_token_limit(
    text: str, max_tokens: int = 512, encoding_name: str = "cl100k_base"
) -> str:
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
            f"Text trimmed from {current_tokens} to {get_token_count(trimmed_text, encoding_name)} tokens (limit: {max_tokens})"
        )
        return trimmed_text

    except Exception as e:
        logger.warning(f"Failed to trim with tiktoken: {e}. Using character approximation.")
        # Fallback to character approximation
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        trimmed = text[: max_chars - 3] + "..."
        logger.warning(
            f"Text trimmed from {len(text)} to {len(trimmed)} characters (token limit: {max_tokens})"
        )
        return trimmed


def encode_documents(
    texts: List[str],
    model: SentenceTransformer,
    max_tokens: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> List[List[float]]:
    """
    Encode documents for storage in vector database.

    Args:
        texts: List of document texts to encode
        model: Loaded SentenceTransformer model
        max_tokens: Optional token limit per document (will trim if specified)
        encoding_name: Tiktoken encoding to use for token counting

    Returns:
        List of embedding vectors (one per document)
    """
    if not texts:
        return []

    # Trim texts if token limit specified
    processed_texts = texts
    if max_tokens:
        processed_texts = [
            trim_text_to_token_limit(text, max_tokens, encoding_name) for text in texts
        ]

    try:
        # Generate embeddings for documents
        # For Stella models, documents don't need prompts (they're the "docs")
        embeddings = model.encode(processed_texts)
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
            logger.debug(f"Using Arctic-Embed query prefix")

        # For Stella models, use s2p_query prompt for sentence-to-passage search
        elif "stella" in model_name:
            try:
                embedding = model.encode([processed_query], prompt_name="s2p_query")[0]
                return embedding.tolist()
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(
                    f"Failed to use s2p_query prompt for Stella model: {e}. Using default encoding."
                )

        # For Qwen3 models, use query prompt for better search performance
        elif "qwen" in model_name:
            try:
                embedding = model.encode([processed_query], prompt_name="query")[0]
                return embedding.tolist()
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(
                    f"Failed to use query prompt for Qwen model: {e}. Using default encoding."
                )

        # Standard encoding without prompt (fallback for all models)
        embedding = model.encode([processed_query])[0]
        return embedding.tolist()

    except Exception as e:
        logger.error(f"Failed to encode query '{query}': {e}")
        raise


def validate_embedding_dimensions(
    embeddings: List[List[float]], expected_dim: Optional[int] = None
) -> bool:
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
