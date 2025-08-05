"""Content analyzer for extracting metadata from markdown sections."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .reference_scanner import ReferenceScanner
from .section_splitter import SectionData


@dataclass
class ContentMetadata:
    """Content analysis metadata."""

    # Content characteristics
    word_count: int = 0
    has_code_blocks: bool = False
    has_tables: bool = False
    has_images: bool = False

    # References and links
    cross_doc_refs: List[str] = None
    external_links: List[str] = None
    internal_links: List[str] = None

    # Document context (from frontmatter)
    document_title: str = ""
    document_tags: List[str] = None
    document_category: str = ""

    # Content integrity
    content_hash: str = ""

    def __post_init__(self):
        if self.cross_doc_refs is None:
            self.cross_doc_refs = []
        if self.external_links is None:
            self.external_links = []
        if self.internal_links is None:
            self.internal_links = []
        if self.document_tags is None:
            self.document_tags = []


class ContentAnalyzer:
    """Analyzes markdown content to extract metadata."""

    def __init__(self):
        self.reference_scanner = ReferenceScanner()

    def _ensure_json_serializable(self, obj: Any) -> Any:
        """Recursively ensure all values in an object are JSON serializable and ChromaDB compatible."""
        if isinstance(obj, dict):
            # Convert None values to empty strings but keep the keys
            return {
                key: self._ensure_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                self._ensure_json_serializable(item) for item in obj if item is not None
            ]
        elif obj is None:
            return ""  # Convert None to empty string for ChromaDB compatibility
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Convert non-serializable types to strings
                return str(obj)

    def analyze_content(
        self,
        section: SectionData,
        frontmatter: Dict[str, Any] = None,
        source_file: str = None,
    ) -> ContentMetadata:
        """
        Analyze section content and extract metadata.

        Args:
            section: Section data to analyze
            frontmatter: Document frontmatter data
            source_file: Source file path for reference resolution

        Returns:
            ContentMetadata with analysis results
        """
        metadata = ContentMetadata()

        # Analyze content characteristics
        metadata.word_count = self.reference_scanner.count_words(section.content)
        metadata.has_code_blocks = self.reference_scanner.has_code_blocks(
            section.content
        )
        metadata.has_tables = self.reference_scanner.has_tables(section.content)
        metadata.has_images = (
            len(self.reference_scanner.find_image_references(section.content)) > 0
        )

        # Analyze references and links
        refs = self.reference_scanner.scan_references(section.content, source_file)
        metadata.cross_doc_refs = refs["cross_doc_refs"]
        metadata.external_links = refs["external_links"]
        metadata.internal_links = refs["internal_links"]

        # Extract document-level metadata from frontmatter
        if frontmatter:
            metadata.document_title = str(frontmatter.get("title", ""))

            # Handle tags - could be list or string
            tags = frontmatter.get("tags", [])
            if isinstance(tags, str):
                metadata.document_tags = [tags]
            elif isinstance(tags, list):
                metadata.document_tags = [str(tag) for tag in tags]

            metadata.document_category = str(frontmatter.get("category", ""))

        # Generate content hash for change detection
        metadata.content_hash = self._generate_content_hash(section.content)

        return metadata

    def analyze_document_context(
        self, frontmatter: Dict[str, Any] = None, source_file: str = None
    ) -> Dict[str, Any]:
        """
        Extract document-level context metadata.

        Args:
            frontmatter: Document frontmatter
            source_file: Source file path

        Returns:
            Document context metadata
        """
        context = {}

        if frontmatter:
            # Extract common frontmatter fields
            context["document_title"] = str(frontmatter.get("title", ""))
            context["document_author"] = str(frontmatter.get("author", ""))
            context["document_description"] = str(frontmatter.get("description", ""))
            context["document_category"] = str(frontmatter.get("category", ""))

            # Handle dates
            for date_field in ["created", "published", "updated"]:
                if date_field in frontmatter:
                    context[f"document_{date_field}"] = str(frontmatter[date_field])

            # Handle tags
            tags = frontmatter.get("tags", [])
            if isinstance(tags, str):
                context["document_tags"] = [tags]
            elif isinstance(tags, list):
                context["document_tags"] = [str(tag) for tag in tags]
            else:
                context["document_tags"] = []
        else:
            # Default values when no frontmatter
            context = {
                "document_title": "",
                "document_author": "",
                "document_description": "",
                "document_category": "",
                "document_tags": [],
            }

        # Extract category from file path if not in frontmatter
        if not context["document_category"] and source_file:
            context["document_category"] = self._extract_category_from_path(source_file)

        return self._ensure_json_serializable(context)

    def _extract_category_from_path(self, file_path: str) -> str:
        """Extract category from file path."""
        from pathlib import Path

        try:
            path = Path(file_path)
            # Use the parent directory name as category
            if path.parent.name and path.parent.name != ".":
                return path.parent.name
        except Exception:
            pass

        return ""

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def enrich_section_metadata(
        self,
        section: SectionData,
        navigation_meta: Dict[str, Any],
        content_meta: ContentMetadata,
        document_context: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
    ) -> Dict[str, Any]:
        """
        Combine all metadata into final section metadata.

        Args:
            section: Section data
            navigation_meta: Navigation metadata
            content_meta: Content analysis metadata
            document_context: Document-level context
            chunk_index: Index of this chunk in document
            total_chunks: Total chunks in document

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            # Basic section info
            "type": self._get_section_type(section),
            "source_file": "",  # Will be set by caller
            "section_level": section.header.level if section.header else 0,
            "title": (
                section.header.text
                if section.header
                else document_context.get("document_title", "")
            ),
            # Navigation metadata
            "section_path": navigation_meta.get("section_path", ""),
            "previous_chunk": navigation_meta.get("previous_chunk"),
            "next_chunk": navigation_meta.get("next_chunk"),
            "parent_section": navigation_meta.get("parent_section"),
            "child_sections": navigation_meta.get("child_sections", []),
            "sibling_sections": navigation_meta.get("sibling_sections", []),
            # Content analysis
            "cross_doc_refs": content_meta.cross_doc_refs,
            "external_links": content_meta.external_links,
            "internal_links": content_meta.internal_links,
            "word_count": content_meta.word_count,
            "has_code_blocks": content_meta.has_code_blocks,
            "has_tables": content_meta.has_tables,
            "has_images": content_meta.has_images,
            # Document context
            "frontmatter": {},  # Will be set by caller if needed
            **document_context,
            # Processing metadata
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "content_hash": content_meta.content_hash,
        }

        # Add split-specific metadata if applicable
        if section.is_split_section:
            metadata["is_split_section"] = True
            metadata["split_index"] = section.split_index
            metadata["total_splits"] = section.total_splits
            metadata["split_parent_id"] = section.split_parent_id
            if section.parent_context:
                metadata["hierarchical_context"] = section.parent_context
        else:
            metadata["is_split_section"] = False

        return self._ensure_json_serializable(metadata)

    def _get_section_type(self, section: SectionData) -> str:
        """Get the type string for a section."""
        if section.section_type == "frontmatter":
            return "markdown_frontmatter"
        elif section.section_type == "document":
            return "markdown_document"
        else:
            return "markdown_section"

    def analyze_batch(
        self,
        sections: List[SectionData],
        frontmatter: Dict[str, Any] = None,
        source_file: str = None,
    ) -> List[ContentMetadata]:
        """
        Analyze multiple sections in batch.

        Args:
            sections: List of sections to analyze
            frontmatter: Document frontmatter
            source_file: Source file path

        Returns:
            List of ContentMetadata for each section
        """
        results = []

        for section in sections:
            metadata = self.analyze_content(section, frontmatter, source_file)
            results.append(metadata)

        return results

    def validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Validate metadata for completeness and consistency.

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = [
            "type",
            "source_file",
            "section_level",
            "title",
            "word_count",
            "chunk_index",
            "total_chunks",
            "content_hash",
        ]

        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")

        # Check data types
        if "word_count" in metadata and not isinstance(metadata["word_count"], int):
            errors.append("word_count must be an integer")

        if "section_level" in metadata and not isinstance(
            metadata["section_level"], int
        ):
            errors.append("section_level must be an integer")

        # Check list fields
        list_fields = [
            "cross_doc_refs",
            "external_links",
            "internal_links",
            "child_sections",
            "sibling_sections",
            "document_tags",
        ]

        for field in list_fields:
            if field in metadata and not isinstance(metadata[field], list):
                errors.append(f"{field} must be a list")

        # Check boolean fields
        bool_fields = [
            "has_code_blocks",
            "has_tables",
            "has_images",
            "is_split_section",
        ]

        for field in bool_fields:
            if field in metadata and not isinstance(metadata[field], bool):
                errors.append(f"{field} must be a boolean")

        return errors
