"""Utilities for converting metadata between OpenAPI format and ChromaDB format."""

import json
from typing import Any, Dict, Union


def prepare_metadata_for_chromadb(
    metadata: Dict[str, Any],
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Convert complex metadata to ChromaDB-compatible format.

    ChromaDB only accepts scalar values (string, int, float, bool) in metadata.
    This function converts complex types (dict, list) to JSON strings.

    Args:
        metadata: Original metadata dictionary with potentially complex values

    Returns:
        ChromaDB-compatible metadata with complex values as JSON strings
    """
    chromadb_metadata = {}

    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            # Convert complex types to JSON strings
            chromadb_metadata[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            # Keep scalars as-is (None becomes None in ChromaDB)
            chromadb_metadata[key] = value
        else:
            # Convert other types to string representation
            chromadb_metadata[key] = str(value)

    return chromadb_metadata


def restore_metadata_from_chromadb(chromadb_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ChromaDB metadata back to original format.

    This function attempts to parse JSON strings back to their original
    complex types (dict, list) while keeping scalars unchanged.

    Args:
        chromadb_metadata: Metadata from ChromaDB with JSON strings

    Returns:
        Restored metadata with complex types reconstructed
    """
    original_metadata = {}

    # Fields that should be parsed as JSON (originally were complex types)
    json_fields = {
        "ref_ids",  # dict mapping chunk IDs to their dependencies
        "referenced_by",  # list of chunk IDs that reference this chunk
        "status_codes",  # list of HTTP status codes (for error responses)
        "tags",  # list of OpenAPI tags for operations
    }

    for key, value in chromadb_metadata.items():
        if key in json_fields and isinstance(value, str):
            try:
                # Try to parse as JSON
                original_metadata[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, keep as string
                original_metadata[key] = value
        else:
            # Keep scalars unchanged
            original_metadata[key] = value

    return original_metadata


def validate_metadata_roundtrip(original_metadata: Dict[str, Any]) -> bool:
    """
    Validate that metadata can be converted to ChromaDB format and back without loss.

    Args:
        original_metadata: Original metadata to test

    Returns:
        True if roundtrip conversion preserves the data, False otherwise
    """
    try:
        # Convert to ChromaDB format
        chromadb_format = prepare_metadata_for_chromadb(original_metadata)

        # Convert back to original format
        restored_format = restore_metadata_from_chromadb(chromadb_format)

        # Compare original and restored
        return original_metadata == restored_format

    except Exception:
        return False
