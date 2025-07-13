"""Tests for Reference Resolver."""

from src.openapi_processor.reference_resolver import ReferenceResolver


class TestReferenceResolver:
    """Test cases for ReferenceResolver."""

    def test_resolve_schema_reference(self):
        """Test resolving schema reference."""
        resolver = ReferenceResolver()
        chunk_id = resolver.resolve_ref_to_chunk_id("#/components/schemas/Pet", "api.json")

        assert chunk_id == "api.json:components/schemas/Pet"

    def test_resolve_parameter_reference(self):
        """Test resolving parameter reference."""
        resolver = ReferenceResolver()
        chunk_id = resolver.resolve_ref_to_chunk_id(
            "#/components/parameters/limitParam", "petstore.yaml"
        )

        assert chunk_id == "petstore.yaml:components/parameters/limitParam"

    def test_resolve_response_reference(self):
        """Test resolving response reference."""
        resolver = ReferenceResolver()
        chunk_id = resolver.resolve_ref_to_chunk_id(
            "#/components/responses/ErrorResponse", "apis/v1/openapi.json"
        )

        assert chunk_id == "apis/v1/openapi.json:components/responses/ErrorResponse"

    def test_resolve_request_body_reference(self):
        """Test resolving request body reference."""
        resolver = ReferenceResolver()
        chunk_id = resolver.resolve_ref_to_chunk_id(
            "#/components/requestBodies/PetBody", "test.yaml"
        )

        assert chunk_id == "test.yaml:components/requestBodies/PetBody"

    def test_resolve_invalid_references(self):
        """Test resolving invalid references."""
        resolver = ReferenceResolver()

        # Non-internal reference
        assert resolver.resolve_ref_to_chunk_id("external.json#/schema", "api.json") is None

        # Empty or None reference
        assert resolver.resolve_ref_to_chunk_id("", "api.json") is None
        assert resolver.resolve_ref_to_chunk_id(None, "api.json") is None

        # Non-string reference
        assert resolver.resolve_ref_to_chunk_id(123, "api.json") is None

    def test_resolve_deep_path_reference(self):
        """Test resolving reference with deep path."""
        resolver = ReferenceResolver()
        chunk_id = resolver.resolve_ref_to_chunk_id(
            "#/components/schemas/nested/deep/Component", "complex.json"
        )

        assert chunk_id == "complex.json:components/schemas/nested/deep/Component"

    def test_different_spec_names(self):
        """Test with different spec name formats."""
        resolver = ReferenceResolver()

        # Simple filename
        assert (
            resolver.resolve_ref_to_chunk_id("#/components/schemas/Pet", "api.json")
            == "api.json:components/schemas/Pet"
        )

        # Path with directories
        assert (
            resolver.resolve_ref_to_chunk_id("#/components/schemas/Pet", "apis/v2/petstore.yaml")
            == "apis/v2/petstore.yaml:components/schemas/Pet"
        )
