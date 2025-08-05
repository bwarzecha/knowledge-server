"""Navigation builder for creating sequential and hierarchical relationships."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .section_splitter import SectionData


@dataclass
class NavigationMetadata:
    """Navigation metadata for a section."""

    # Sequential navigation
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None

    # Hierarchical relationships
    parent_section: Optional[str] = None
    child_sections: List[str] = None
    sibling_sections: List[str] = None

    # Section path and context
    section_path: str = ""
    hierarchical_context: str = ""

    def __post_init__(self):
        if self.child_sections is None:
            self.child_sections = []
        if self.sibling_sections is None:
            self.sibling_sections = []


class NavigationBuilder:
    """Builds navigation relationships between markdown sections."""

    def __init__(self):
        pass

    def build_navigation(
        self, sections: List[SectionData], filename: str
    ) -> Dict[str, NavigationMetadata]:
        """
        Build navigation metadata for all sections.

        Args:
            sections: List of section data objects
            filename: Source filename for generating chunk IDs

        Returns:
            Dictionary mapping section identifiers to navigation metadata
        """
        if not sections:
            return {}

        # Generate chunk IDs for all sections
        section_ids = []
        section_lookup = {}

        for i, section in enumerate(sections):
            chunk_id = self._generate_chunk_id(section, filename, i)
            section_ids.append(chunk_id)
            section_lookup[chunk_id] = section

        navigation = {}

        # Build sequential navigation (previous/next)
        for i, section_id in enumerate(section_ids):
            nav_meta = NavigationMetadata()

            # Set previous/next
            if i > 0:
                nav_meta.previous_chunk = section_ids[i - 1]
            if i < len(section_ids) - 1:
                nav_meta.next_chunk = section_ids[i + 1]

            navigation[section_id] = nav_meta

        # Build hierarchical relationships
        self._build_hierarchical_navigation(
            sections, section_ids, section_lookup, navigation
        )

        # Build section paths and context
        self._build_section_paths(sections, section_ids, section_lookup, navigation)

        return navigation

    def _generate_chunk_id(
        self, section: SectionData, filename: str, index: int
    ) -> str:
        """
        Generate chunk ID for a section.

        Args:
            section: Section data
            filename: Source filename
            index: Section index

        Returns:
            Chunk ID string
        """
        if section.section_type == "frontmatter":
            return f"{filename}:frontmatter"
        elif section.section_type == "document":
            return f"{filename}:document"
        elif section.header:
            section_identifier = self._slugify(section.header.text)
            # Add line number to ensure uniqueness
            line_num = section.header.line_number
            # Handle split sections
            if (
                section.is_split_section
                and section.split_index
                and section.split_index > 1
            ):
                return (
                    f"{filename}:{section_identifier}:L{line_num}-{section.split_index}"
                )
            return f"{filename}:{section_identifier}:L{line_num}"
        else:
            # Fallback
            return f"{filename}:section-{index}"

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        import re

        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s_]+", "-", slug)
        return slug.strip("-")

    def _build_hierarchical_navigation(
        self,
        sections: List[SectionData],
        section_ids: List[str],
        section_lookup: Dict[str, SectionData],
        navigation: Dict[str, NavigationMetadata],
    ) -> None:
        """Build parent/child/sibling relationships."""

        # Create mapping of header levels to sections
        header_sections = []
        for i, section in enumerate(sections):
            if section.header:
                header_sections.append(
                    {
                        "index": i,
                        "section_id": section_ids[i],
                        "section": section,
                        "level": section.header.level,
                    }
                )

        if not header_sections:
            return

        # Build parent-child relationships
        for i, current in enumerate(header_sections):
            current_level = current["level"]
            current_id = current["section_id"]
            nav_meta = navigation[current_id]

            # Find parent (most recent header at lower level)
            parent_id = None
            for j in range(i - 1, -1, -1):
                prev_section = header_sections[j]
                if prev_section["level"] < current_level:
                    parent_id = prev_section["section_id"]
                    break

            if parent_id:
                nav_meta.parent_section = parent_id
                # Add current as child to parent
                parent_nav = navigation[parent_id]
                if current_id not in parent_nav.child_sections:
                    parent_nav.child_sections.append(current_id)

        # Build sibling relationships
        for i, current in enumerate(header_sections):
            current_level = current["level"]
            current_id = current["section_id"]
            nav_meta = navigation[current_id]

            # Find siblings (same level, same parent)
            current_parent = nav_meta.parent_section

            for other in header_sections:
                if other["section_id"] == current_id:
                    continue

                other_id = other["section_id"]
                other_nav = navigation[other_id]

                # Same level and same parent = siblings
                if (
                    other["level"] == current_level
                    and other_nav.parent_section == current_parent
                ):
                    if other_id not in nav_meta.sibling_sections:
                        nav_meta.sibling_sections.append(other_id)

    def _build_section_paths(
        self,
        sections: List[SectionData],
        section_ids: List[str],
        section_lookup: Dict[str, SectionData],
        navigation: Dict[str, NavigationMetadata],
    ) -> None:
        """Build section paths and hierarchical context strings."""

        for section_id, nav_meta in navigation.items():
            section = section_lookup[section_id]

            if section.section_type in ["frontmatter", "document"]:
                nav_meta.section_path = ""
                nav_meta.hierarchical_context = ""
                continue

            if not section.header:
                continue

            # Build hierarchical path by walking up parent chain
            path_parts = []
            context_parts = []
            current_id = section_id

            while current_id:
                current_section = section_lookup[current_id]
                if current_section.header:
                    path_parts.insert(0, self._slugify(current_section.header.text))
                    context_parts.insert(0, current_section.header.text)

                # Move to parent
                current_nav = navigation[current_id]
                current_id = current_nav.parent_section

            nav_meta.section_path = "/".join(path_parts)
            nav_meta.hierarchical_context = " > ".join(context_parts)

    def build_cross_document_navigation(
        self, all_files_sections: Dict[str, List[SectionData]]
    ) -> Dict[str, NavigationMetadata]:
        """
        Build navigation across multiple documents.

        Args:
            all_files_sections: Dictionary mapping filenames to their sections

        Returns:
            Complete navigation metadata for all sections across all files
        """
        all_navigation = {}

        # Build navigation for each file independently
        for filename, sections in all_files_sections.items():
            file_navigation = self.build_navigation(sections, filename)
            all_navigation.update(file_navigation)

        # Here we could add cross-document navigation logic if needed
        # For now, each document is self-contained

        return all_navigation

    def enhance_split_section_navigation(
        self, sections: List[SectionData], navigation: Dict[str, NavigationMetadata]
    ) -> None:
        """
        Enhance navigation for split sections to maintain relationships.

        Args:
            sections: List of section data
            navigation: Navigation metadata to enhance
        """
        # Group sections by split parent
        split_groups = {}
        for i, section in enumerate(sections):
            if section.is_split_section and section.split_parent_id:
                parent_id = section.split_parent_id
                if parent_id not in split_groups:
                    split_groups[parent_id] = []
                split_groups[parent_id].append(i)

        # Update navigation for split sections
        for parent_id, section_indices in split_groups.items():
            if len(section_indices) <= 1:
                continue

            # Sort by split index
            section_indices.sort(key=lambda i: sections[i].split_index or 0)

            # Update parent-child relationships for split sections
            for j, section_idx in enumerate(section_indices):
                section = sections[section_idx]
                # The first split section is the "parent" for navigation purposes
                if j == 0:
                    # This is the main section - others are its children
                    continue
                else:
                    # This is a split child - update its parent reference
                    # (The parent is the first split section, not the logical parent)
                    pass

    def validate_navigation(
        self, navigation: Dict[str, NavigationMetadata]
    ) -> List[str]:
        """
        Validate navigation relationships for consistency.

        Args:
            navigation: Navigation metadata to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for section_id, nav_meta in navigation.items():
            # Check that previous/next references exist
            if nav_meta.previous_chunk and nav_meta.previous_chunk not in navigation:
                errors.append(
                    f"Section {section_id} references non-existent previous chunk {nav_meta.previous_chunk}"
                )

            if nav_meta.next_chunk and nav_meta.next_chunk not in navigation:
                errors.append(
                    f"Section {section_id} references non-existent next chunk {nav_meta.next_chunk}"
                )

            # Check that parent references exist
            if nav_meta.parent_section and nav_meta.parent_section not in navigation:
                errors.append(
                    f"Section {section_id} references non-existent parent {nav_meta.parent_section}"
                )

            # Check that child references exist
            for child_id in nav_meta.child_sections:
                if child_id not in navigation:
                    errors.append(
                        f"Section {section_id} references non-existent child {child_id}"
                    )

            # Check that sibling references exist
            for sibling_id in nav_meta.sibling_sections:
                if sibling_id not in navigation:
                    errors.append(
                        f"Section {section_id} references non-existent sibling {sibling_id}"
                    )

        return errors
