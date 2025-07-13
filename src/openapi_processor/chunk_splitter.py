"""Adaptive chunk splitter for handling oversized OpenAPI elements."""

from typing import List

import yaml

from .extractor import ExtractedElement


class ChunkSplitter:
    """Splits oversized chunks into smaller, embedding-friendly pieces."""

    def __init__(self, max_chunk_size: int = 8000):
        """
        Initialize the chunk splitter.

        Args:
            max_chunk_size: Maximum characters per chunk (default: 8000)
        """
        self.max_chunk_size = max_chunk_size

    def split_element(self, element: ExtractedElement) -> List[ExtractedElement]:
        """
        Split an element based on type and size.

        Args:
            element: The element to split

        Returns:
            List of elements (original or split)
        """
        if element.element_type == "operation":
            # Always split operations - error codes are noise
            return self._split_operation(element)

        elif element.element_type == "component":
            # Only split if too large
            if self._is_too_large(element):
                return self._split_component(element)
            else:
                return [element]

        else:
            # Other types unchanged
            return [element]

    def _is_too_large(self, element: ExtractedElement) -> bool:
        """Check if an element exceeds the size threshold."""
        test_chunk = yaml.dump(element.content, default_flow_style=False)
        return len(test_chunk) > self.max_chunk_size

    def _split_component(self, element: ExtractedElement) -> List[ExtractedElement]:
        """Split a component schema, separating examples from definition."""
        result = []

        # Get the schema name and content
        schema_name = list(element.content.keys())[0]
        schema_def = element.content[schema_name]

        # Create main chunk with core definition (no examples)
        main_def = {}
        examples_data = {}

        for key, value in schema_def.items():
            if key == "example":
                examples_data["example"] = value
            elif key == "examples":
                examples_data["examples"] = value
            else:
                main_def[key] = value

        # Create the main definition chunk (replaces original)
        main_metadata = element.metadata.copy()
        main_metadata["chunk_type"] = "definition"

        # Add reference to examples if they exist
        if examples_data:
            examples_chunk_id = f"{element.element_id}:examples"
            main_metadata["has_examples"] = examples_chunk_id
            # Add to ref_ids for graph builder to create bidirectional references
            if "ref_ids" not in main_metadata:
                main_metadata["ref_ids"] = {}
            main_metadata["ref_ids"][examples_chunk_id] = []

        main_chunk = ExtractedElement(
            element_id=element.element_id,
            element_type=element.element_type,
            content={schema_name: main_def},
            metadata=main_metadata,
        )
        result.append(main_chunk)

        # Create separate example chunk if examples exist
        if examples_data:
            example_metadata = element.metadata.copy()
            example_metadata["chunk_type"] = "examples"
            example_metadata["parent_definition"] = element.element_id
            # Reset ref_ids and only add parent reference
            example_metadata["ref_ids"] = {element.element_id: []}

            example_chunk = ExtractedElement(
                element_id=f"{element.element_id}:examples",
                element_type=element.element_type,
                content={f"{schema_name}_examples": examples_data},
                metadata=example_metadata,
            )
            result.append(example_chunk)

        return result

    def _split_operation(self, element: ExtractedElement) -> List[ExtractedElement]:
        """Split an operation, always separating error responses from main operation."""
        result = []

        # Get operation method and content
        operation_method = list(element.content.keys())[0]
        operation_def = element.content[operation_method]

        # Split responses into success (200-299) and error (400+) codes
        main_responses = {}
        error_responses = {}

        if "responses" in operation_def:
            for status_code, response_def in operation_def["responses"].items():
                # Parse status code - handle both string and int
                try:
                    status_num = int(str(status_code))
                    if 200 <= status_num <= 299:
                        main_responses[status_code] = response_def
                    else:
                        error_responses[status_code] = response_def
                except ValueError:
                    # For non-numeric status codes (like 'default'), put in error responses
                    error_responses[status_code] = response_def

        # Create main operation chunk with success responses only (replaces original)
        main_operation = {k: v for k, v in operation_def.items() if k != "responses"}
        if main_responses:
            main_operation["responses"] = main_responses

        main_metadata = element.metadata.copy()
        main_metadata["chunk_type"] = "operation"

        # Add reference to error responses if they exist
        if error_responses:
            error_chunk_id = f"{element.element_id}:errors"
            main_metadata["error_responses"] = error_chunk_id
            # Add to ref_ids for graph builder to create bidirectional references
            if "ref_ids" not in main_metadata:
                main_metadata["ref_ids"] = {}
            main_metadata["ref_ids"][error_chunk_id] = []

        main_chunk = ExtractedElement(
            element_id=element.element_id,
            element_type=element.element_type,
            content={operation_method: main_operation},
            metadata=main_metadata,
        )
        result.append(main_chunk)

        # Create error responses chunk if error responses exist
        if error_responses:
            error_metadata = element.metadata.copy()
            error_metadata["chunk_type"] = "error_responses"
            error_metadata["parent_operation"] = element.element_id
            # Reset ref_ids and only add parent reference
            error_metadata["ref_ids"] = {element.element_id: []}

            error_chunk = ExtractedElement(
                element_id=f"{element.element_id}:errors",
                element_type=element.element_type,
                content={f"{operation_method}_error_responses": error_responses},
                metadata=error_metadata,
            )
            result.append(error_chunk)

        return result
