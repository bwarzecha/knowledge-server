"""Tests for Chunk Assembler."""

import yaml

from src.openapi_processor.chunk_assembler import ChunkAssembler
from src.openapi_processor.extractor import ExtractedElement


class TestChunkAssembler:
    """Test cases for ChunkAssembler."""

    def test_assemble_info_chunk(self):
        """Test assembling an info chunk."""
        element = ExtractedElement(
            element_id="api.json:info",
            element_type="info",
            content={"title": "Test API", "version": "1.0.0"},
            metadata={
                "type": "info",
                "source_file": "api.json",
                "natural_name": "info",
                "ref_ids": {},
                "referenced_by": [],
            },
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        # Check structure
        assert "id" in chunk
        assert "document" in chunk
        assert "metadata" in chunk

        # Check ID
        assert chunk["id"] == "api.json:info"

        # Check document is valid YAML
        document_data = yaml.safe_load(chunk["document"])
        assert document_data == {"title": "Test API", "version": "1.0.0"}

        # Check metadata
        metadata = chunk["metadata"]
        assert metadata["type"] == "info"
        assert metadata["source_file"] == "api.json"
        assert metadata["api_info_ref"] == "api.json:info"
        assert metadata["api_tags_ref"] == "api.json:tags"

    def test_assemble_operation_chunk(self):
        """Test assembling an operation chunk."""
        element = ExtractedElement(
            element_id="api.json:paths/pets/get",
            element_type="operation",
            content={
                "get": {
                    "operationId": "listPets",
                    "responses": {"200": {"description": "Success"}},
                }
            },
            metadata={
                "type": "operation",
                "source_file": "api.json",
                "path": "/pets",
                "method": "get",
                "operation_id": "listPets",
                "tags": ["pets"],
                "ref_ids": {"api.json:components/schemas/Pet": []},
                "referenced_by": [],
            },
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        assert chunk["id"] == "api.json:paths/pets/get"

        # Check YAML document
        document_data = yaml.safe_load(chunk["document"])
        assert "get" in document_data
        assert document_data["get"]["operationId"] == "listPets"

        # Check operation-specific metadata
        metadata = chunk["metadata"]
        assert metadata["path"] == "/pets"
        assert metadata["method"] == "get"
        assert metadata["operation_id"] == "listPets"
        assert metadata["tags"] == ["pets"]

    def test_assemble_component_chunk(self):
        """Test assembling a component chunk."""
        element = ExtractedElement(
            element_id="api.json:components/schemas/Pet",
            element_type="component",
            content={
                "Pet": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                }
            },
            metadata={
                "type": "component",
                "source_file": "api.json",
                "component_type": "schemas",
                "component_name": "Pet",
                "ref_ids": {},
                "referenced_by": ["api.json:paths/pets/get"],
            },
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        assert chunk["id"] == "api.json:components/schemas/Pet"

        # Check YAML document
        document_data = yaml.safe_load(chunk["document"])
        assert "Pet" in document_data
        assert document_data["Pet"]["type"] == "object"

        # Check component-specific metadata
        metadata = chunk["metadata"]
        assert metadata["component_type"] == "schemas"
        assert metadata["component_name"] == "Pet"
        assert "api.json:paths/pets/get" in metadata["referenced_by"]

    def test_assemble_chunk_with_references(self):
        """Test assembling chunk with complex references."""
        element = ExtractedElement(
            element_id="api.json:paths/pets/post",
            element_type="operation",
            content={
                "post": {
                    "requestBody": {"$ref": "#/components/requestBodies/PetBody"},
                    "responses": {
                        "200": {"$ref": "#/components/responses/PetResponse"},
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            metadata={
                "type": "operation",
                "source_file": "api.json",
                "path": "/pets",
                "method": "post",
                "ref_ids": {
                    "api.json:components/requestBodies/PetBody": [
                        "api.json:components/schemas/Pet"
                    ],
                    "api.json:components/responses/PetResponse": [],
                    "api.json:components/responses/ErrorResponse": [],
                },
                "referenced_by": [],
            },
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        # Check references are preserved in metadata
        ref_ids = chunk["metadata"]["ref_ids"]
        assert "api.json:components/requestBodies/PetBody" in ref_ids
        assert "api.json:components/responses/PetResponse" in ref_ids
        assert "api.json:components/responses/ErrorResponse" in ref_ids

        # Check hierarchical dependencies
        pet_body_deps = ref_ids["api.json:components/requestBodies/PetBody"]
        assert "api.json:components/schemas/Pet" in pet_body_deps

    def test_assemble_chunk_missing_metadata(self):
        """Test assembling chunk with missing metadata fields."""
        element = ExtractedElement(
            element_id="api.json:components/schemas/Simple",
            element_type="component",
            content={"Simple": {"type": "string"}},
            metadata={
                "type": "component",
                "source_file": "api.json",
                "component_name": "Simple",
                # Missing ref_ids, referenced_by, natural_name
            },
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        # Should fill in missing fields
        metadata = chunk["metadata"]
        assert "ref_ids" in metadata
        assert "referenced_by" in metadata
        assert "natural_name" in metadata
        assert metadata["ref_ids"] == {}
        assert metadata["referenced_by"] == []
        assert metadata["natural_name"] == "Simple"

    def test_yaml_formatting(self):
        """Test YAML formatting is readable and correct."""
        element = ExtractedElement(
            element_id="test:component",
            element_type="component",
            content={
                "ComplexSchema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Item"},
                        },
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["items"],
                }
            },
            metadata={"type": "component", "source_file": "test.json"},
        )

        assembler = ChunkAssembler()
        chunk = assembler.assemble_chunk(element)

        document = chunk["document"]

        # Should be valid YAML
        parsed = yaml.safe_load(document)
        assert parsed is not None

        # Should preserve structure
        assert "ComplexSchema" in parsed
        assert "properties" in parsed["ComplexSchema"]
        assert "required" in parsed["ComplexSchema"]

        # Should be readable (not flow style)
        assert "items:" in document  # Block style
        assert not document.startswith("{")  # Not flow style
