"""Complex integration tests for Graph Builder with real OpenAPI samples."""

from src.openapi_processor.extractor import ElementExtractor
from src.openapi_processor.graph_builder import GraphBuilder
from src.openapi_processor.parser import OpenAPIParser
from src.openapi_processor.validator import OpenAPIValidator


class TestGraphBuilderComplex:
    """Test Graph Builder with complex real-world OpenAPI samples."""

    def test_complex_nesting_dependency_tree(self):
        """Test dependency tree for complex-nesting.json."""
        # Parse and process the file
        parser = OpenAPIParser()
        validator = OpenAPIValidator()
        extractor = ElementExtractor()
        builder = GraphBuilder()

        parse_result = parser.parse_file("open-api-small-samples/3.0/json/complex-nesting.json")
        assert parse_result.success

        validation_result = validator.validate(parse_result.data)
        assert validation_result.is_valid

        elements = extractor.extract_elements(parse_result.data, "complex-nesting.json")
        result = builder.build_reference_graph(elements)

        # Build actual dependency tree
        actual_tree = {}
        for element in result:
            actual_tree[element.element_id] = {
                "refs": element.metadata["ref_ids"],
                "referenced_by": set(element.metadata["referenced_by"]),
            }

        # Define expected dependency tree for key schemas
        # ObjectOfEverything references 3 schemas, each with their own dependencies
        expected_tree = {
            "complex-nesting.json:components/schemas/ObjectOfEverything": {
                "refs": {
                    "complex-nesting.json:components/schemas/ObjectOfObjectsAndArrays": [
                        "complex-nesting.json:components/schemas/FlatObject",
                        "complex-nesting.json:components/schemas/ArrayOfPrimitives",
                        "complex-nesting.json:components/schemas/ArrayOfFlatObjects",
                    ],
                    "complex-nesting.json:components/schemas/ArrayOfObjectsOfObjectsAndArrays": [
                        "complex-nesting.json:components/schemas/ObjectOfObjectsAndArrays",
                        "complex-nesting.json:components/schemas/FlatObject",
                        "complex-nesting.json:components/schemas/ArrayOfPrimitives",
                        "complex-nesting.json:components/schemas/ArrayOfFlatObjects",
                    ],
                    "complex-nesting.json:components/schemas/ObjectOfAdditionalPropertiesObjectPolymorphism": [
                        "complex-nesting.json:components/schemas/FlatObject",
                        "complex-nesting.json:components/schemas/ObjectWithArray",
                    ],
                }
            },
            "complex-nesting.json:components/schemas/ObjectOfObjectsAndArrays": {
                "refs": {
                    "complex-nesting.json:components/schemas/FlatObject": [],
                    "complex-nesting.json:components/schemas/ArrayOfPrimitives": [],
                    "complex-nesting.json:components/schemas/ArrayOfFlatObjects": [
                        "complex-nesting.json:components/schemas/FlatObject"
                    ],
                }
            },
            "complex-nesting.json:components/schemas/ArrayOfFlatObjects": {
                "refs": {"complex-nesting.json:components/schemas/FlatObject": []}
            },
            "complex-nesting.json:components/schemas/FlatObject": {
                "refs": {},  # Leaf node
                "referenced_by": {
                    "complex-nesting.json:components/schemas/ObjectOfObjectsAndArrays",
                    "complex-nesting.json:components/schemas/ArrayOfFlatObjects",
                    # Plus any operations that use it
                },
            },
        }

        # Verify key parts match expected
        for schema_id in expected_tree:
            if schema_id in actual_tree:
                actual = actual_tree[schema_id]
                expected = expected_tree[schema_id]

                # Compare refs - keys should match exactly
                assert actual["refs"].keys() == expected["refs"].keys()
                # Compare dependency lists as sets since order doesn't matter
                for ref_id, deps in actual["refs"].items():
                    assert set(deps) == set(expected["refs"][ref_id])

                # For referenced_by, just check subset since there may be more references from operations
                if "referenced_by" in expected:
                    assert expected["referenced_by"].issubset(actual["referenced_by"])

    def test_circular_reference_tree(self):
        """Test circular reference handling with a clear expected tree."""
        # Create test elements with circular references: A -> B -> C -> A
        elements = [
            {
                "id": "test.json:components/schemas/A",
                "type": "component",
                "content": {"A": {"properties": {"b": {"$ref": "#/components/schemas/B"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
            {
                "id": "test.json:components/schemas/B",
                "type": "component",
                "content": {"B": {"properties": {"c": {"$ref": "#/components/schemas/C"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
            {
                "id": "test.json:components/schemas/C",
                "type": "component",
                "content": {"C": {"properties": {"a": {"$ref": "#/components/schemas/A"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
        ]

        from src.openapi_processor.extractor import ExtractedElement

        extracted = [
            ExtractedElement(
                element_id=e["id"],
                element_type=e["type"],
                content=e["content"],
                metadata=e["metadata"],
            )
            for e in elements
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(extracted)

        # Expected: Each schema includes its full dependency chain
        expected_refs = {
            "test.json:components/schemas/A": {
                "test.json:components/schemas/B": [
                    "test.json:components/schemas/C",
                    "test.json:components/schemas/A",  # B's chain includes C and A
                ]
            },
            "test.json:components/schemas/B": {
                "test.json:components/schemas/C": [
                    "test.json:components/schemas/A",
                    "test.json:components/schemas/B",  # C's chain includes A and B
                ]
            },
            "test.json:components/schemas/C": {
                "test.json:components/schemas/A": [
                    "test.json:components/schemas/B",
                    "test.json:components/schemas/C",  # A's chain includes B and C
                ]
            },
        }

        # Verify the circular dependency tree
        for element in result:
            expected = expected_refs[element.element_id]
            actual_refs = element.metadata["ref_ids"]

            # Compare structure
            assert actual_refs.keys() == expected.keys()
            # Compare dependencies as sets
            for ref_id, deps in actual_refs.items():
                assert set(deps) == set(expected[ref_id])

    def test_diamond_dependency_pattern(self):
        """Test diamond pattern: A -> B,C; B -> D; C -> D (D is referenced via two paths)."""
        elements = [
            {
                "id": "test.json:schemas/D",
                "content": {"D": {"type": "object"}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
            {
                "id": "test.json:schemas/B",
                "content": {"B": {"properties": {"d": {"$ref": "#/schemas/D"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
            {
                "id": "test.json:schemas/C",
                "content": {"C": {"properties": {"d": {"$ref": "#/schemas/D"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
            {
                "id": "test.json:schemas/A",
                "content": {"A": {"properties": {"b": {"$ref": "#/schemas/B"}, "c": {"$ref": "#/schemas/C"}}}},
                "metadata": {"type": "component", "source_file": "test.json"},
            },
        ]

        from src.openapi_processor.extractor import ExtractedElement

        extracted = [
            ExtractedElement(
                element_id=e["id"],
                element_type="component",
                content=e["content"],
                metadata=e["metadata"],
            )
            for e in elements
        ]

        builder = GraphBuilder()
        result = builder.build_reference_graph(extracted)

        # Expected dependency tree
        expected = {
            "test.json:schemas/A": {
                "ref_ids": {
                    "test.json:schemas/B": ["test.json:schemas/D"],
                    "test.json:schemas/C": ["test.json:schemas/D"],
                },
                "referenced_by": [],
            },
            "test.json:schemas/B": {
                "ref_ids": {"test.json:schemas/D": []},
                "referenced_by": ["test.json:schemas/A"],
            },
            "test.json:schemas/C": {
                "ref_ids": {"test.json:schemas/D": []},
                "referenced_by": ["test.json:schemas/A"],
            },
            "test.json:schemas/D": {
                "ref_ids": {},
                "referenced_by": ["test.json:schemas/B", "test.json:schemas/C"],
            },
        }

        # Build actual result map
        actual = {}
        for element in result:
            actual[element.element_id] = {
                "ref_ids": element.metadata["ref_ids"],
                "referenced_by": set(element.metadata["referenced_by"]),
            }

        # Verify the diamond pattern
        assert actual["test.json:schemas/A"]["ref_ids"] == expected["test.json:schemas/A"]["ref_ids"]
        assert actual["test.json:schemas/D"]["referenced_by"] == set(expected["test.json:schemas/D"]["referenced_by"])

    def test_deep_linear_chain(self):
        """Test deep linear dependency chain: A -> B -> C -> D -> E."""
        # Build a chain of schemas
        schemas = ["E", "D", "C", "B", "A"]  # Build from leaf to root
        elements = []

        for i, schema in enumerate(schemas):
            if i == 0:  # E is the leaf
                content = {schema: {"type": "string"}}
            else:  # Others reference the previous
                prev = schemas[i - 1]
                content = {schema: {"properties": {"ref": {"$ref": f"#/schemas/{prev}"}}}}

            from src.openapi_processor.extractor import ExtractedElement

            elements.append(
                ExtractedElement(
                    element_id=f"test.json:schemas/{schema}",
                    element_type="component",
                    content=content,
                    metadata={"type": "component", "source_file": "test.json"},
                )
            )

        builder = GraphBuilder()
        result = builder.build_reference_graph(elements)

        # Expected: Each level includes all transitive dependencies
        expected_tree = {
            "test.json:schemas/A": {
                "ref_ids": {
                    "test.json:schemas/B": [
                        "test.json:schemas/C",
                        "test.json:schemas/D",
                        "test.json:schemas/E",
                    ]
                }
            },
            "test.json:schemas/B": {"ref_ids": {"test.json:schemas/C": ["test.json:schemas/D", "test.json:schemas/E"]}},
            "test.json:schemas/C": {"ref_ids": {"test.json:schemas/D": ["test.json:schemas/E"]}},
            "test.json:schemas/D": {"ref_ids": {"test.json:schemas/E": []}},
            "test.json:schemas/E": {"ref_ids": {}},
        }

        # Verify the chain
        for element in result:
            expected = expected_tree[element.element_id]
            actual_refs = element.metadata["ref_ids"]

            # Compare structure
            assert actual_refs.keys() == expected["ref_ids"].keys()
            # Compare dependencies as sets
            for ref_id, deps in actual_refs.items():
                assert set(deps) == set(expected["ref_ids"][ref_id])
