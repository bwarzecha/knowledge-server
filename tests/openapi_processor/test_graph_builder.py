"""Tests for Graph Builder."""


from src.openapi_processor.extractor import ExtractedElement
from src.openapi_processor.graph_builder import GraphBuilder


class TestGraphBuilder:
    """Test cases for GraphBuilder."""

    def test_build_simple_reference_graph(self):
        """Test building graph with simple references."""
        # Create test elements
        elements = [
            ExtractedElement(
                element_id="api.json:components/schemas/Pet",
                element_type="component",
                content={"Pet": {"type": "object", "properties": {"name": {"type": "string"}}}},
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:paths/pets/get",
                element_type="operation",
                content={
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Pet"}
                                    }
                                }
                            }
                        }
                    }
                },
                metadata={"type": "operation", "source_file": "api.json"},
            ),
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected graph structure
        expected_refs = {
            "api.json:paths/pets/get": {
                "ref_ids": {"api.json:components/schemas/Pet": []},
                "referenced_by": [],
            },
            "api.json:components/schemas/Pet": {
                "ref_ids": {},
                "referenced_by": ["api.json:paths/pets/get"],
            },
        }

        # Verify the graph matches expected structure
        for element in result:
            expected = expected_refs[element.element_id]
            assert element.metadata["ref_ids"] == expected["ref_ids"]
            assert set(element.metadata["referenced_by"]) == set(expected["referenced_by"])

    def test_build_no_references(self):
        """Test building graph with no references."""
        elements = [
            ExtractedElement(
                element_id="api.json:info",
                element_type="info",
                content={"title": "Test API", "version": "1.0.0"},
                metadata={"type": "info", "source_file": "api.json"},
            )
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        element = result[0]
        assert element.metadata["ref_ids"] == {}
        assert element.metadata["referenced_by"] == []

    def test_build_transitive_references(self):
        """Test building graph with transitive references (A -> B -> C)."""
        elements = [
            ExtractedElement(
                element_id="api.json:components/schemas/Address",
                element_type="component",
                content={"Address": {"type": "object"}},
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/Person",
                element_type="component",
                content={
                    "Person": {
                        "type": "object",
                        "properties": {"address": {"$ref": "#/components/schemas/Address"}},
                    }
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:paths/people/get",
                element_type="operation",
                content={
                    "get": {
                        "responses": {"200": {"schema": {"$ref": "#/components/schemas/Person"}}}
                    }
                },
                metadata={"type": "operation", "source_file": "api.json"},
            ),
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected graph structure
        expected_refs = {
            "api.json:paths/people/get": {
                "ref_ids": {
                    "api.json:components/schemas/Person": ["api.json:components/schemas/Address"]
                },
                "referenced_by": [],
            },
            "api.json:components/schemas/Person": {
                "ref_ids": {"api.json:components/schemas/Address": []},
                "referenced_by": ["api.json:paths/people/get"],
            },
            "api.json:components/schemas/Address": {
                "ref_ids": {},
                "referenced_by": ["api.json:components/schemas/Person"],  # Only direct references
            },
        }

        # Verify the graph
        for element in result:
            expected = expected_refs[element.element_id]
            assert element.metadata["ref_ids"] == expected["ref_ids"]
            assert set(element.metadata["referenced_by"]) == set(expected["referenced_by"])

    def test_circular_reference_handling(self):
        """Test handling of circular references (A -> B -> A)."""
        elements = [
            ExtractedElement(
                element_id="api.json:components/schemas/A",
                element_type="component",
                content={
                    "A": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/B"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/B",
                element_type="component",
                content={
                    "B": {"type": "object", "properties": {"a": {"$ref": "#/components/schemas/A"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected: Each references the other, with A included in B's deps (visited is copied)
        expected_refs = {
            "api.json:components/schemas/A": {
                "ref_ids": {
                    "api.json:components/schemas/B": [
                        "api.json:components/schemas/A"
                    ]  # B has A as dependency
                },
                "referenced_by": ["api.json:components/schemas/B"],
            },
            "api.json:components/schemas/B": {
                "ref_ids": {
                    "api.json:components/schemas/A": [
                        "api.json:components/schemas/B"
                    ]  # A has B as dependency
                },
                "referenced_by": ["api.json:components/schemas/A"],
            },
        }

        for element in result:
            expected = expected_refs[element.element_id]
            assert element.metadata["ref_ids"] == expected["ref_ids"]
            assert set(element.metadata["referenced_by"]) == set(expected["referenced_by"])

    def test_deep_nested_dependencies(self):
        """Test deep dependency chain: Operation -> Company -> Person -> Address -> BaseType."""
        elements = [
            ExtractedElement(
                element_id="api.json:components/schemas/BaseType",
                element_type="component",
                content={"BaseType": {"type": "string"}},
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/Address",
                element_type="component",
                content={
                    "Address": {"properties": {"type": {"$ref": "#/components/schemas/BaseType"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/Person",
                element_type="component",
                content={
                    "Person": {"properties": {"address": {"$ref": "#/components/schemas/Address"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/Company",
                element_type="component",
                content={
                    "Company": {"properties": {"owner": {"$ref": "#/components/schemas/Person"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:paths/companies/get",
                element_type="operation",
                content={
                    "get": {
                        "responses": {"200": {"schema": {"$ref": "#/components/schemas/Company"}}}
                    }
                },
                metadata={"type": "operation", "source_file": "api.json"},
            ),
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected graph structure - each level includes all transitive dependencies
        expected_refs = {
            "api.json:paths/companies/get": {
                "ref_ids": {
                    "api.json:components/schemas/Company": [
                        "api.json:components/schemas/Person",
                        "api.json:components/schemas/Address",
                        "api.json:components/schemas/BaseType",
                    ]
                }
            },
            "api.json:components/schemas/Company": {
                "ref_ids": {
                    "api.json:components/schemas/Person": [
                        "api.json:components/schemas/Address",
                        "api.json:components/schemas/BaseType",
                    ]
                }
            },
            "api.json:components/schemas/Person": {
                "ref_ids": {
                    "api.json:components/schemas/Address": ["api.json:components/schemas/BaseType"]
                }
            },
            "api.json:components/schemas/Address": {
                "ref_ids": {"api.json:components/schemas/BaseType": []}
            },
            "api.json:components/schemas/BaseType": {"ref_ids": {}},
        }

        # Verify ref_ids match expected (order may differ in lists)
        for element in result:
            expected = expected_refs[element.element_id]
            # Compare ref_ids keys
            assert element.metadata["ref_ids"].keys() == expected["ref_ids"].keys()
            # Compare dependency lists (order doesn't matter)
            for ref_id, deps in element.metadata["ref_ids"].items():
                assert set(deps) == set(expected["ref_ids"][ref_id])

    def test_multiple_paths_to_same_dependency(self):
        """Test when multiple references lead to the same element."""
        elements = [
            ExtractedElement(
                element_id="api.json:components/schemas/Common",
                element_type="component",
                content={"Common": {"type": "object"}},
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/TypeA",
                element_type="component",
                content={
                    "TypeA": {"properties": {"common": {"$ref": "#/components/schemas/Common"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:components/schemas/TypeB",
                element_type="component",
                content={
                    "TypeB": {"properties": {"common": {"$ref": "#/components/schemas/Common"}}}
                },
                metadata={"type": "component", "source_file": "api.json"},
            ),
            ExtractedElement(
                element_id="api.json:paths/test/post",
                element_type="operation",
                content={
                    "post": {
                        "requestBody": {"schema": {"$ref": "#/components/schemas/TypeA"}},
                        "responses": {"200": {"schema": {"$ref": "#/components/schemas/TypeB"}}},
                    }
                },
                metadata={"type": "operation", "source_file": "api.json"},
            ),
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected: Operation references both TypeA and TypeB, each includes Common
        expected_refs = {
            "api.json:paths/test/post": {
                "ref_ids": {
                    "api.json:components/schemas/TypeA": ["api.json:components/schemas/Common"],
                    "api.json:components/schemas/TypeB": ["api.json:components/schemas/Common"],
                }
            },
            "api.json:components/schemas/Common": {
                "referenced_by": [
                    "api.json:components/schemas/TypeA",
                    "api.json:components/schemas/TypeB",
                    # Operation doesn't directly reference Common
                ]
            },
        }

        # Verify operation's references
        operation = next(e for e in result if e.element_type == "operation")
        assert operation.metadata["ref_ids"] == expected_refs["api.json:paths/test/post"]["ref_ids"]

        # Verify Common is referenced by all three
        common = next(e for e in result if e.element_id == "api.json:components/schemas/Common")
        assert set(common.metadata["referenced_by"]) == set(
            expected_refs["api.json:components/schemas/Common"]["referenced_by"]
        )

    def test_invalid_references(self):
        """Test handling of references to non-existent elements."""
        elements = [
            ExtractedElement(
                element_id="api.json:paths/pets/get",
                element_type="operation",
                content={
                    "get": {
                        "responses": {
                            "200": {"schema": {"$ref": "#/components/schemas/NonExistent"}}
                        }
                    }
                },
                metadata={"type": "operation", "source_file": "api.json"},
            )
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        operation = result[0]
        # Should handle gracefully - no references to non-existent elements
        assert operation.metadata["ref_ids"] == {}
        assert operation.metadata["referenced_by"] == []
