"""Tests for metadata conversion utilities."""


from src.vector_store.metadata_utils import (
    prepare_metadata_for_chromadb,
    restore_metadata_from_chromadb,
    validate_metadata_roundtrip,
)


class TestMetadataUtils:
    """Test metadata conversion between OpenAPI format and ChromaDB format."""

    def test_prepare_simple_metadata(self):
        """Test conversion of simple scalar metadata."""
        metadata = {
            "source_file": "api.json",
            "chunk_type": "definition",
            "element_type": "component",
            "line_count": 42,
        }

        result = prepare_metadata_for_chromadb(metadata)

        # Simple scalars should remain unchanged
        assert result == metadata

    def test_prepare_complex_metadata(self):
        """Test conversion of complex metadata with dict and list values."""
        metadata = {
            "source_file": "api.json",
            "ref_ids": {
                "api.json:User": ["api.json:Address", "api.json:Phone"],
                "api.json:Company": [],
            },
            "referenced_by": ["api.json:paths/users/get", "api.json:paths/users/post"],
            "chunk_type": "definition",
        }

        result = prepare_metadata_for_chromadb(metadata)

        # Scalars should remain unchanged
        assert result["source_file"] == "api.json"
        assert result["chunk_type"] == "definition"

        # Complex types should become JSON strings
        assert isinstance(result["ref_ids"], str)
        assert isinstance(result["referenced_by"], str)

        # JSON strings should be valid
        import json

        parsed_ref_ids = json.loads(result["ref_ids"])
        assert parsed_ref_ids == metadata["ref_ids"]

        parsed_referenced_by = json.loads(result["referenced_by"])
        assert parsed_referenced_by == metadata["referenced_by"]

    def test_restore_simple_metadata(self):
        """Test restoration of simple scalar metadata."""
        chromadb_metadata = {
            "source_file": "api.json",
            "chunk_type": "definition",
            "element_type": "component",
        }

        result = restore_metadata_from_chromadb(chromadb_metadata)

        # Should remain unchanged
        assert result == chromadb_metadata

    def test_restore_complex_metadata(self):
        """Test restoration of complex metadata from JSON strings."""
        import json

        original_ref_ids = {"api.json:User": ["api.json:Address"], "api.json:Company": []}
        original_referenced_by = ["api.json:paths/users/get"]

        chromadb_metadata = {
            "source_file": "api.json",
            "ref_ids": json.dumps(original_ref_ids),
            "referenced_by": json.dumps(original_referenced_by),
            "chunk_type": "definition",
        }

        result = restore_metadata_from_chromadb(chromadb_metadata)

        # Scalars should remain unchanged
        assert result["source_file"] == "api.json"
        assert result["chunk_type"] == "definition"

        # JSON fields should be parsed back to original types
        assert result["ref_ids"] == original_ref_ids
        assert result["referenced_by"] == original_referenced_by

    def test_roundtrip_conversion(self):
        """Test that metadata survives roundtrip conversion unchanged."""
        original_metadata = {
            "source_file": "api.json",
            "element_type": "component",
            "chunk_type": "definition",
            "ref_ids": {
                "api.json:User": ["api.json:Address", "api.json:Phone"],
                "api.json:Company": [],
            },
            "referenced_by": ["api.json:paths/users/get", "api.json:paths/users/post"],
            "has_examples": "api.json:User:examples",
            "natural_name": "User",
        }

        # Convert to ChromaDB format
        chromadb_format = prepare_metadata_for_chromadb(original_metadata)

        # Convert back to original format
        restored_format = restore_metadata_from_chromadb(chromadb_format)

        # Should match exactly
        assert restored_format == original_metadata

    def test_validate_metadata_roundtrip(self):
        """Test the roundtrip validation function."""
        # Valid metadata that should survive roundtrip
        valid_metadata = {
            "source_file": "api.json",
            "ref_ids": {"api.json:User": ["api.json:Address"]},
            "referenced_by": ["api.json:Operation1"],
            "chunk_type": "definition",
        }

        assert validate_metadata_roundtrip(valid_metadata) is True

        # Test with simple metadata (no complex types)
        simple_metadata = {"source_file": "api.json", "chunk_type": "definition"}

        assert validate_metadata_roundtrip(simple_metadata) is True

    def test_handle_none_values(self):
        """Test handling of None values in metadata."""
        metadata = {"source_file": "api.json", "optional_field": None, "chunk_type": "definition"}

        chromadb_format = prepare_metadata_for_chromadb(metadata)
        restored_format = restore_metadata_from_chromadb(chromadb_format)

        assert restored_format == metadata

    def test_handle_invalid_json_in_restore(self):
        """Test graceful handling of invalid JSON strings during restoration."""
        chromadb_metadata = {
            "source_file": "api.json",
            "ref_ids": "invalid json string {",  # Invalid JSON
            "chunk_type": "definition",
        }

        result = restore_metadata_from_chromadb(chromadb_metadata)

        # Invalid JSON should be kept as string
        assert result["ref_ids"] == "invalid json string {"
        assert result["source_file"] == "api.json"
        assert result["chunk_type"] == "definition"
