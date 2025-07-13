"""Chunk Assembler for creating final chunks with YAML formatting."""

from typing import Any, Dict, List

import yaml

from .chunk_splitter import ChunkSplitter
from .extractor import ExtractedElement


class ChunkAssembler:
    """Assembles final chunks with YAML documents and complete metadata."""

    def __init__(self):
        self.splitter = ChunkSplitter()

    def assemble_chunks(self, elements: List[ExtractedElement]) -> List[Dict[str, Any]]:
        """
        Assemble multiple elements into final chunks, applying splitting as needed.

        Args:
            elements: List of ExtractedElement objects

        Returns:
            List of final chunk dictionaries
        """
        # Phase 1: Split all elements
        all_split_elements = []
        for element in elements:
            split_elements = self.splitter.split_element(element)
            all_split_elements.extend(split_elements)

        # Phase 2: Update referenced_by for split elements
        self._populate_referenced_by_for_splits(all_split_elements)

        # Phase 3: Assemble final chunks
        final_chunks = []
        for split_element in all_split_elements:
            chunk = self.assemble_chunk(split_element)
            final_chunks.append(chunk)

        return final_chunks

    def assemble_chunk(self, element: ExtractedElement) -> Dict[str, Any]:
        """
        Create final chunk with YAML document and complete metadata.

        Args:
            element: ExtractedElement with content and metadata

        Returns:
            Complete chunk dictionary matching the contract
        """
        # Convert content to YAML format
        document = self._convert_to_yaml(element.content)

        # Build complete metadata
        metadata = self._build_complete_metadata(element)

        return {"id": element.element_id, "document": document, "metadata": metadata}

    def _populate_referenced_by_for_splits(self, elements: List[ExtractedElement]) -> None:
        """
        Populate referenced_by lists for split elements based on their ref_ids.
        Similar to GraphBuilder._populate_referenced_by but for post-split elements.
        """
        # Create lookup map
        element_lookup = {element.element_id: element for element in elements}

        # Clear all referenced_by lists first
        for element in elements:
            element.metadata["referenced_by"] = []

        # Populate referenced_by based on ref_ids
        for element in elements:
            ref_ids = element.metadata.get("ref_ids", {})
            for referenced_chunk_id in ref_ids.keys():
                if referenced_chunk_id in element_lookup:
                    referenced_element = element_lookup[referenced_chunk_id]
                    if element.element_id not in referenced_element.metadata["referenced_by"]:
                        referenced_element.metadata["referenced_by"].append(element.element_id)

    def _convert_to_yaml(self, content: Any) -> str:
        """Convert content to YAML format."""
        try:
            return yaml.dump(
                content, default_flow_style=False, sort_keys=False, allow_unicode=True
            ).strip()
        except Exception:
            # Fallback for content that can't be serialized
            return str(content)

    def _build_complete_metadata(self, element: ExtractedElement) -> Dict[str, Any]:
        """Build complete metadata for the chunk."""
        metadata = element.metadata.copy()

        # Ensure required fields exist
        if "ref_ids" not in metadata:
            metadata["ref_ids"] = {}
        if "referenced_by" not in metadata:
            metadata["referenced_by"] = []

        # Add API context references
        source_file = metadata["source_file"]
        metadata["api_info_ref"] = f"{source_file}:info"

        # Add tags reference if it might exist (we'll validate later)
        metadata["api_tags_ref"] = f"{source_file}:tags"

        # Ensure natural_name exists
        if "natural_name" not in metadata:
            metadata["natural_name"] = self._extract_natural_name(element)

        return metadata

    def _extract_natural_name(self, element: ExtractedElement) -> str:
        """Extract natural name from element."""
        if element.element_type == "info":
            return "info"
        elif element.element_type == "tags":
            return "tags"
        elif element.element_type == "operation":
            path = element.metadata.get("path", "")
            method = element.metadata.get("method", "")
            return f"{path}/{method}"
        elif element.element_type == "component":
            return element.metadata.get("component_name", element.element_id.split(":")[-1])
        else:
            # Fallback to last part of element ID
            return element.element_id.split(":")[-1]
