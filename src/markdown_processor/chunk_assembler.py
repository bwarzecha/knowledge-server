"""Chunk assembler for creating final chunks with complete metadata."""

import json
from typing import Any, Dict, List

from .content_analyzer import ContentAnalyzer
from .navigation_builder import NavigationBuilder
from .section_splitter import SectionData


class ChunkAssembler:
    """Assembles final chunks from sections with complete metadata."""

    def __init__(self):
        self.navigation_builder = NavigationBuilder()
        self.content_analyzer = ContentAnalyzer()

    def _ensure_json_serializable(self, obj: Any) -> Any:
        """Recursively ensure all values in an object are JSON serializable and ChromaDB compatible."""
        if isinstance(obj, dict):
            # Convert None values to empty strings but keep the keys
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj if item is not None]
        elif obj is None:
            return ""  # Convert None to empty string for ChromaDB compatibility
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Convert non-serializable types to strings
                return str(obj)

    def assemble_chunks(
        self,
        sections: List[SectionData],
        frontmatter: Dict[str, Any] = None,
        source_file: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Assemble final chunks from sections.

        Args:
            sections: List of section data objects
            frontmatter: Document frontmatter
            source_file: Source file path

        Returns:
            List of complete chunk dictionaries ready for vector storage
        """
        if not sections:
            return []

        chunks = []

        # Build navigation metadata for all sections
        navigation_data = self.navigation_builder.build_navigation(sections, source_file or "unknown")

        # Analyze document context once
        document_context = self.content_analyzer.analyze_document_context(frontmatter, source_file)

        # Process each section
        for i, section in enumerate(sections):
            chunk = self._assemble_single_chunk(
                section=section,
                frontmatter=frontmatter,
                navigation_data=navigation_data,
                document_context=document_context,
                source_file=source_file or "unknown",
                chunk_index=i,
                total_chunks=len(sections),
            )
            chunks.append(chunk)

        return chunks

    def _assemble_single_chunk(
        self,
        section: SectionData,
        frontmatter: Dict[str, Any],
        navigation_data: Dict[str, Any],
        document_context: Dict[str, Any],
        source_file: str,
        chunk_index: int,
        total_chunks: int,
    ) -> Dict[str, Any]:
        """
        Assemble a single chunk with complete metadata.

        Args:
            section: Section data
            frontmatter: Document frontmatter
            navigation_data: Navigation metadata for all sections
            document_context: Document-level context
            source_file: Source file path
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Complete chunk dictionary
        """
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(section, source_file, chunk_index)

        # Get navigation metadata for this section
        nav_meta = navigation_data.get(chunk_id, {})
        if hasattr(nav_meta, "__dict__"):
            nav_meta = nav_meta.__dict__

        # Analyze content
        content_meta = self.content_analyzer.analyze_content(section, frontmatter, source_file)

        # Build complete metadata
        metadata = self.content_analyzer.enrich_section_metadata(
            section=section,
            navigation_meta=nav_meta,
            content_meta=content_meta,
            document_context=document_context,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

        # Set source file in metadata
        metadata["source_file"] = source_file

        # Add frontmatter to metadata if this is frontmatter chunk
        if section.section_type == "frontmatter" and frontmatter:
            metadata["frontmatter"] = self._ensure_json_serializable(frontmatter)
        elif frontmatter:
            # For other chunks, include relevant frontmatter fields
            metadata["frontmatter"] = self._ensure_json_serializable(
                {
                    "title": frontmatter.get("title"),
                    "author": frontmatter.get("author"),
                    "tags": frontmatter.get("tags", []),
                    "category": frontmatter.get("category"),
                }
            )
        else:
            metadata["frontmatter"] = {}

        # Create final chunk
        chunk = {"id": chunk_id, "document": section.content, "metadata": metadata}

        return chunk

    def _generate_chunk_id(self, section: SectionData, source_file: str, index: int) -> str:
        """
        Generate chunk ID for a section.

        Args:
            section: Section data
            source_file: Source file path
            index: Section index

        Returns:
            Chunk ID string
        """
        # Use the same logic as navigation builder
        return self.navigation_builder._generate_chunk_id(section, source_file, index)

    def assemble_batch(self, files_sections: Dict[str, tuple]) -> List[Dict[str, Any]]:
        """
        Assemble chunks for multiple files in batch.

        Args:
            files_sections: Dictionary mapping filenames to (sections, frontmatter) tuples

        Returns:
            List of all chunks from all files
        """
        all_chunks = []

        for source_file, (sections, frontmatter) in files_sections.items():
            file_chunks = self.assemble_chunks(sections, frontmatter, source_file)
            all_chunks.extend(file_chunks)

        return all_chunks

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Validate assembled chunks for completeness and consistency.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not chunks:
            return errors

        chunk_ids = set()

        for i, chunk in enumerate(chunks):
            # Check chunk structure
            if not isinstance(chunk, dict):
                errors.append(f"Chunk {i} is not a dictionary")
                continue

            # Check required top-level fields
            required_fields = ["id", "document", "metadata"]
            for field in required_fields:
                if field not in chunk:
                    errors.append(f"Chunk {i} missing required field: {field}")

            # Check chunk ID uniqueness
            chunk_id = chunk.get("id")
            if chunk_id:
                if chunk_id in chunk_ids:
                    errors.append(f"Duplicate chunk ID: {chunk_id}")
                chunk_ids.add(chunk_id)

            # Validate metadata
            if "metadata" in chunk:
                metadata_errors = self.content_analyzer.validate_metadata(chunk["metadata"])
                for error in metadata_errors:
                    errors.append(f"Chunk {chunk_id or i} metadata error: {error}")

        # Validate navigation relationships
        nav_errors = self._validate_navigation_consistency(chunks)
        errors.extend(nav_errors)

        return errors

    def _validate_navigation_consistency(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Validate navigation relationships between chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of navigation validation errors
        """
        errors = []
        chunk_ids = {chunk.get("id") for chunk in chunks if chunk.get("id")}

        for chunk in chunks:
            chunk_id = chunk.get("id")
            metadata = chunk.get("metadata", {})

            # Check previous/next references
            prev_chunk = metadata.get("previous_chunk")
            if prev_chunk and prev_chunk not in chunk_ids:
                errors.append(f"Chunk {chunk_id} references non-existent previous chunk: {prev_chunk}")

            next_chunk = metadata.get("next_chunk")
            if next_chunk and next_chunk not in chunk_ids:
                errors.append(f"Chunk {chunk_id} references non-existent next chunk: {next_chunk}")

            # Check parent/child references
            parent_section = metadata.get("parent_section")
            if parent_section and parent_section not in chunk_ids:
                errors.append(f"Chunk {chunk_id} references non-existent parent: {parent_section}")

            child_sections = metadata.get("child_sections", [])
            for child_id in child_sections:
                if child_id not in chunk_ids:
                    errors.append(f"Chunk {chunk_id} references non-existent child: {child_id}")

            # Check sibling references
            sibling_sections = metadata.get("sibling_sections", [])
            for sibling_id in sibling_sections:
                if sibling_id not in chunk_ids:
                    errors.append(f"Chunk {chunk_id} references non-existent sibling: {sibling_id}")

        return errors

    def optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunks for better search performance.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Optimized chunks
        """
        optimized = []

        for chunk in chunks:
            optimized_chunk = chunk.copy()

            # Clean up metadata - remove empty or null values
            metadata = optimized_chunk.get("metadata", {})
            cleaned_metadata = {}

            for key, value in metadata.items():
                if value is not None and value != "" and value != []:
                    cleaned_metadata[key] = value

            optimized_chunk["metadata"] = cleaned_metadata

            # Ensure document content is clean
            document = optimized_chunk.get("document", "")
            if isinstance(document, str):
                # Remove excessive whitespace but preserve structure
                lines = document.split("\n")
                cleaned_lines = []
                prev_empty = False

                for line in lines:
                    if line.strip():
                        cleaned_lines.append(line.rstrip())
                        prev_empty = False
                    else:
                        # Only keep one consecutive empty line
                        if not prev_empty:
                            cleaned_lines.append("")
                        prev_empty = True

                optimized_chunk["document"] = "\n".join(cleaned_lines).strip()

            optimized.append(optimized_chunk)

        return optimized
