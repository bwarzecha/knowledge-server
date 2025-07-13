"""Tests for Element Extractor."""

from src.openapi_processor.extractor import ElementExtractor
from src.openapi_processor.parser import OpenAPIParser


class TestElementExtractor:
    """Test cases for ElementExtractor."""

    def test_extract_minimal_spec(self):
        """Test extracting elements from a minimal OpenAPI spec."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "test.json")

        # Should extract info element only (no tags, no operations, no components)
        assert len(elements) == 1

        info_element = elements[0]
        assert info_element.element_type == "info"
        assert info_element.element_id == "test.json:info"
        assert info_element.content == spec_data["info"]
        assert info_element.metadata["type"] == "info"
        assert info_element.metadata["source_file"] == "test.json"

    def test_extract_spec_with_tags(self):
        """Test extracting spec with tags section."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "tags": [
                {"name": "pets", "description": "Pet operations"},
                {"name": "store", "description": "Store operations"},
            ],
            "paths": {},
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "test.yaml")

        # Should extract info and tags
        assert len(elements) == 2

        tags_element = next(e for e in elements if e.element_type == "tags")
        assert tags_element.element_id == "test.yaml:tags"
        assert tags_element.content == spec_data["tags"]
        assert tags_element.metadata["type"] == "tags"

    def test_extract_operations(self):
        """Test extracting operations from paths."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "tags": ["pets"],
                        "summary": "List all pets",
                    },
                    "post": {
                        "operationId": "createPet",
                        "tags": ["pets"],
                        "summary": "Create a pet",
                    },
                },
                "/pets/{id}": {
                    "get": {"operationId": "getPet", "tags": ["pets"], "summary": "Get a pet by ID"}
                },
            },
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "api.json")

        # Should extract info + 3 operations
        assert len(elements) == 4

        operation_elements = [e for e in elements if e.element_type == "operation"]
        assert len(operation_elements) == 3

        # Check specific operations
        get_pets = next(
            e
            for e in operation_elements
            if e.metadata["path"] == "/pets" and e.metadata["method"] == "get"
        )
        assert get_pets.element_id == "api.json:paths/pets/get"
        assert get_pets.metadata["operation_id"] == "listPets"
        assert get_pets.metadata["tags"] == ["pets"]
        assert "get" in get_pets.content

        post_pets = next(
            e
            for e in operation_elements
            if e.metadata["path"] == "/pets" and e.metadata["method"] == "post"
        )
        assert post_pets.element_id == "api.json:paths/pets/post"
        assert post_pets.metadata["operation_id"] == "createPet"

    def test_extract_components(self):
        """Test extracting components."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "schemas": {
                    "Pet": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                    },
                    "Error": {
                        "type": "object",
                        "properties": {"code": {"type": "integer"}, "message": {"type": "string"}},
                    },
                },
                "parameters": {
                    "limitParam": {"name": "limit", "in": "query", "schema": {"type": "integer"}}
                },
            },
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "api.yaml")

        # Should extract info + 3 components (2 schemas + 1 parameter)
        assert len(elements) == 4

        component_elements = [e for e in elements if e.element_type == "component"]
        assert len(component_elements) == 3

        # Check schema components
        schema_elements = [
            e for e in component_elements if e.metadata["component_type"] == "schemas"
        ]
        assert len(schema_elements) == 2

        pet_schema = next(e for e in schema_elements if e.metadata["component_name"] == "Pet")
        assert pet_schema.element_id == "api.yaml:components/schemas/Pet"
        assert pet_schema.metadata["component_type"] == "schemas"
        assert "Pet" in pet_schema.content

        # Check parameter component
        param_elements = [
            e for e in component_elements if e.metadata["component_type"] == "parameters"
        ]
        assert len(param_elements) == 1

        param_element = param_elements[0]
        assert param_element.element_id == "api.yaml:components/parameters/limitParam"
        assert param_element.metadata["component_name"] == "limitParam"

    def test_extract_real_petstore_sample(self):
        """Test extracting from real petstore sample file."""
        parser = OpenAPIParser()
        parse_result = parser.parse_file("open-api-small-samples/3.0/json/petstore-simple.json")

        assert parse_result.success is True

        extractor = ElementExtractor()
        elements = extractor.extract_elements(parse_result.data, "petstore-simple.json")

        # Should have info + operations (no tags, no components in this simple version)
        assert len(elements) >= 1  # At least info

        # Check info element
        info_elements = [e for e in elements if e.element_type == "info"]
        assert len(info_elements) == 1

        info_element = info_elements[0]
        assert info_element.metadata["source_file"] == "petstore-simple.json"
        assert "title" in info_element.content

        # Check operations
        operation_elements = [e for e in elements if e.element_type == "operation"]
        assert len(operation_elements) >= 1  # Should have at least one operation

        for op in operation_elements:
            assert op.metadata["path"].startswith("/")
            assert op.metadata["method"] in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "options",
                "head",
                "trace",
            ]
            assert op.element_id.startswith("petstore-simple.json:paths")

    def test_extract_empty_sections(self):
        """Test extraction with empty sections."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "tags": [],  # Empty tags should not create element
            "paths": {},  # Empty paths
            "components": {},  # Empty components
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "empty.json")

        # Should only extract info (empty sections ignored)
        assert len(elements) == 1
        assert elements[0].element_type == "info"

    def test_extract_malformed_sections(self):
        """Test extraction with malformed sections (defensive programming)."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": "not-a-dict",  # Invalid paths
            "components": ["not", "a", "dict"],  # Invalid components
        }

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "malformed.json")

        # Should only extract info (malformed sections skipped)
        assert len(elements) == 1
        assert elements[0].element_type == "info"

    def test_extract_missing_info(self):
        """Test extraction when info section is missing."""
        spec_data = {"openapi": "3.0.0", "paths": {"/test": {"get": {"operationId": "test"}}}}

        extractor = ElementExtractor()
        elements = extractor.extract_elements(spec_data, "no-info.json")

        # Should extract operation but not info
        assert len(elements) == 1
        assert elements[0].element_type == "operation"
