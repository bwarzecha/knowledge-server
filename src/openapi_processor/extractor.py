"""Element Extractor for OpenAPI specifications."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedElement:
    """Represents an extracted element from an OpenAPI specification."""

    element_id: str
    element_type: str  # "info", "tags", "operation", "component"
    content: Dict[str, Any]
    metadata: Dict[str, Any]


class ElementExtractor:
    """Extracts elements from parsed OpenAPI specifications."""

    def extract_elements(self, spec: Dict[str, Any], spec_name: str) -> List[ExtractedElement]:
        """
        Extract all elements from an OpenAPI specification.

        Args:
            spec: Parsed OpenAPI specification data
            spec_name: Relative path with extension (e.g., "apis/v1/openapi.yaml")

        Returns:
            List of ExtractedElement objects
        """
        elements = []

        # Extract info section
        info_element = self._extract_info(spec, spec_name)
        if info_element:
            elements.append(info_element)

        # Extract tags section (if exists)
        tags_element = self._extract_tags(spec, spec_name)
        if tags_element:
            elements.append(tags_element)

        # Extract operations from paths
        operation_elements = self._extract_operations(spec, spec_name)
        elements.extend(operation_elements)

        # Extract components
        component_elements = self._extract_components(spec, spec_name)
        elements.extend(component_elements)

        return elements

    def _extract_info(self, spec: Dict[str, Any], spec_name: str) -> Optional[ExtractedElement]:
        """Extract the info section."""
        if "info" not in spec:
            return None

        element_id = f"{spec_name}:info"
        content = spec["info"]

        metadata = {"type": "info", "source_file": spec_name, "natural_name": "info"}

        return ExtractedElement(
            element_id=element_id, element_type="info", content=content, metadata=metadata
        )

    def _extract_tags(self, spec: Dict[str, Any], spec_name: str) -> Optional[ExtractedElement]:
        """Extract the tags section."""
        if "tags" not in spec or not spec["tags"]:
            return None

        element_id = f"{spec_name}:tags"
        content = spec["tags"]

        metadata = {"type": "tags", "source_file": spec_name, "natural_name": "tags"}

        return ExtractedElement(
            element_id=element_id, element_type="tags", content=content, metadata=metadata
        )

    def _extract_operations(self, spec: Dict[str, Any], spec_name: str) -> List[ExtractedElement]:
        """Extract operations from the paths section."""
        operations = []

        if "paths" not in spec:
            return operations

        paths = spec["paths"]
        if not isinstance(paths, dict):
            return operations

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            # Extract each HTTP method as a separate operation
            http_methods = ["get", "post", "put", "delete", "options", "head", "patch", "trace"]

            for method in http_methods:
                if method in path_item:
                    operation = self._extract_single_operation(
                        path, method, path_item[method], spec_name
                    )
                    if operation:
                        operations.append(operation)

        return operations

    def _extract_single_operation(
        self, path: str, method: str, operation_data: Dict[str, Any], spec_name: str
    ) -> Optional[ExtractedElement]:
        """Extract a single operation."""
        if not isinstance(operation_data, dict):
            return None

        # Create element ID: spec_name:paths{path}/{method}
        element_id = f"{spec_name}:paths{path}/{method}"

        # Content is the operation data wrapped in method key
        content = {method: operation_data}

        # Extract operation metadata
        operation_id = operation_data.get("operationId", "")
        tags = operation_data.get("tags", [])

        metadata = {
            "type": "operation",
            "source_file": spec_name,
            "natural_name": f"{path}/{method}",
            "path": path,
            "method": method,
            "operation_id": operation_id,
            "tags": tags if isinstance(tags, list) else [],
        }

        return ExtractedElement(
            element_id=element_id, element_type="operation", content=content, metadata=metadata
        )

    def _extract_components(self, spec: Dict[str, Any], spec_name: str) -> List[ExtractedElement]:
        """Extract components from the components section."""
        components = []

        if "components" not in spec:
            return components

        components_section = spec["components"]
        if not isinstance(components_section, dict):
            return components

        # Extract each component type (schemas, parameters, responses, etc.)
        for component_type, component_items in components_section.items():
            if not isinstance(component_items, dict):
                continue

            for component_name, component_data in component_items.items():
                component = self._extract_single_component(
                    component_type, component_name, component_data, spec_name
                )
                if component:
                    components.append(component)

        return components

    def _extract_single_component(
        self, component_type: str, component_name: str, component_data: Any, spec_name: str
    ) -> Optional[ExtractedElement]:
        """Extract a single component."""
        # Create element ID: spec_name:components/{type}/{name}
        element_id = f"{spec_name}:components/{component_type}/{component_name}"

        # Content is the component data wrapped with the component name
        content = {component_name: component_data}

        metadata = {
            "type": "component",
            "source_file": spec_name,
            "natural_name": component_name,
            "component_type": component_type,
            "component_name": component_name,
        }

        return ExtractedElement(
            element_id=element_id, element_type="component", content=content, metadata=metadata
        )
