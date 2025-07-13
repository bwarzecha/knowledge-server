"""Tests for Reference Scanner."""

from src.openapi_processor.reference_scanner import ReferenceScanner


class TestReferenceScanner:
    """Test cases for ReferenceScanner."""

    def test_find_no_references(self):
        """Test finding references in content with no $ref."""
        content = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        assert refs == []

    def test_find_single_reference(self):
        """Test finding a single reference."""
        content = {"type": "object", "properties": {"pet": {"$ref": "#/components/schemas/Pet"}}}

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        assert refs == ["#/components/schemas/Pet"]

    def test_find_multiple_references(self):
        """Test finding multiple references."""
        content = {
            "requestBody": {"$ref": "#/components/requestBodies/PetBody"},
            "responses": {
                "200": {"$ref": "#/components/responses/PetResponse"},
                "400": {"$ref": "#/components/responses/ErrorResponse"},
            },
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        expected = [
            "#/components/requestBodies/PetBody",
            "#/components/responses/ErrorResponse",
            "#/components/responses/PetResponse",
        ]
        assert refs == expected

    def test_find_nested_references(self):
        """Test finding references in deeply nested structures."""
        content = {
            "allOf": [
                {"$ref": "#/components/schemas/Base"},
                {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"$ref": "#/components/schemas/Item"}}
                    },
                },
            ]
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        expected = ["#/components/schemas/Base", "#/components/schemas/Item"]
        assert refs == expected

    def test_find_references_in_arrays(self):
        """Test finding references in array structures."""
        content = {
            "oneOf": [{"$ref": "#/components/schemas/Dog"}, {"$ref": "#/components/schemas/Cat"}]
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        expected = ["#/components/schemas/Cat", "#/components/schemas/Dog"]
        assert refs == expected

    def test_deduplicate_references(self):
        """Test that duplicate references are deduplicated."""
        content = {
            "properties": {
                "pet1": {"$ref": "#/components/schemas/Pet"},
                "pet2": {"$ref": "#/components/schemas/Pet"},
            }
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        assert refs == ["#/components/schemas/Pet"]

    def test_ignore_non_string_ref_values(self):
        """Test that non-string $ref values are ignored."""
        content = {
            "$ref": 123,  # Non-string value
            "properties": {"valid": {"$ref": "#/components/schemas/Valid"}},
        }

        scanner = ReferenceScanner()
        refs = scanner.find_references(content)

        assert refs == ["#/components/schemas/Valid"]

    def test_scan_non_dict_content(self):
        """Test scanning non-dictionary content."""
        scanner = ReferenceScanner()

        # Test with string
        assert scanner.find_references("just a string") == []

        # Test with list
        list_content = [
            {"$ref": "#/components/schemas/Item1"},
            {"$ref": "#/components/schemas/Item2"},
        ]
        refs = scanner.find_references(list_content)
        assert refs == ["#/components/schemas/Item1", "#/components/schemas/Item2"]

        # Test with primitive
        assert scanner.find_references(42) == []
        assert scanner.find_references(None) == []
