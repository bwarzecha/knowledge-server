"""Graph Builder for creating bidirectional reference graphs."""

from typing import Dict, List, Set

from .extractor import ExtractedElement
from .reference_resolver import ReferenceResolver
from .reference_scanner import ReferenceScanner


class GraphBuilder:
    """Builds bidirectional reference graphs from extracted elements."""

    def __init__(self):
        self.scanner = ReferenceScanner()
        self.resolver = ReferenceResolver()

    def build_reference_graph(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """
        Build hierarchical ref_ids and referenced_by lists.

        Args:
            elements: List of extracted elements

        Returns:
            List of elements with ref_ids and referenced_by populated
        """
        # Create lookup for all elements by ID
        element_lookup = {elem.element_id: elem for elem in elements}

        # Build forward references (ref_ids)
        for element in elements:
            element.metadata["ref_ids"] = self._build_ref_ids(element, element_lookup)

        # Build backward references (referenced_by)
        for element in elements:
            element.metadata["referenced_by"] = []

        # Populate referenced_by based on ref_ids
        for element in elements:
            self._populate_referenced_by(element, element_lookup)

        return elements

    def _build_ref_ids(
        self, element: ExtractedElement, element_lookup: Dict[str, ExtractedElement]
    ) -> Dict[str, List[str]]:
        """Build hierarchical ref_ids for an element."""
        ref_ids = {}

        # Find all references in element content
        refs = self.scanner.find_references(element.content)

        for ref in refs:
            # Convert $ref to chunk ID
            chunk_id = self.resolver.resolve_ref_to_chunk_id(ref, element.metadata["source_file"])

            if chunk_id and chunk_id in element_lookup:
                # Get dependencies of the referenced element
                dependencies = self._get_dependencies(chunk_id, element_lookup, visited=set())
                ref_ids[chunk_id] = dependencies

        return ref_ids

    def _get_dependencies(
        self, chunk_id: str, element_lookup: Dict[str, ExtractedElement], visited: Set[str]
    ) -> List[str]:
        """Get all dependencies of a chunk (recursive with cycle detection)."""
        if chunk_id in visited or chunk_id not in element_lookup:
            return []

        visited.add(chunk_id)
        dependencies = []

        element = element_lookup[chunk_id]
        refs = self.scanner.find_references(element.content)

        for ref in refs:
            dep_chunk_id = self.resolver.resolve_ref_to_chunk_id(
                ref, element.metadata["source_file"]
            )
            if dep_chunk_id and dep_chunk_id in element_lookup and dep_chunk_id not in visited:
                dependencies.append(dep_chunk_id)
                # Add transitive dependencies
                transitive_deps = self._get_dependencies(
                    dep_chunk_id, element_lookup, visited.copy()
                )
                dependencies.extend(transitive_deps)

        return list(set(dependencies))  # Remove duplicates

    def _populate_referenced_by(
        self, element: ExtractedElement, element_lookup: Dict[str, ExtractedElement]
    ) -> None:
        """Populate referenced_by lists based on ref_ids."""
        ref_ids = element.metadata.get("ref_ids", {})

        for referenced_chunk_id in ref_ids.keys():
            if referenced_chunk_id in element_lookup:
                referenced_element = element_lookup[referenced_chunk_id]
                if element.element_id not in referenced_element.metadata["referenced_by"]:
                    referenced_element.metadata["referenced_by"].append(element.element_id)
