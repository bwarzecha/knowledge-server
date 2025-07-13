"""Adaptive section splitter with token-based chunking."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tiktoken

from .header_extractor import Header


@dataclass
class SplittingConfig:
    """Configuration for adaptive section splitting."""

    max_tokens: int = 1000  # Default 1000, supports up to 8000
    min_tokens: int = 100
    tiktoken_model: str = "gpt-4"
    include_context_in_splits: bool = True
    context_overlap_tokens: int = 50

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_tokens > 8000:
            self.max_tokens = 8000
        elif self.max_tokens < 100:
            self.max_tokens = 100

        if self.min_tokens > self.max_tokens:
            self.min_tokens = min(100, self.max_tokens // 2)

        if self.context_overlap_tokens > self.max_tokens // 4:
            self.context_overlap_tokens = self.max_tokens // 4


@dataclass
class SectionBoundary:
    """Represents a section boundary with start/end positions."""

    header: Optional[Header]
    start_position: int
    end_position: int
    content: str
    section_type: str  # "frontmatter", "document", "section"


@dataclass
class SectionData:
    """Output section data with split metadata."""

    content: str
    header: Optional[Header]
    start_position: int
    end_position: int
    section_type: str
    token_count: int

    # Split metadata
    is_split_section: bool = False
    split_index: Optional[int] = None
    total_splits: Optional[int] = None
    split_parent_id: Optional[str] = None
    parent_context: Optional[str] = None


class AdaptiveSectionSplitter:
    """Splits markdown content into optimally-sized sections using token counting."""

    def __init__(self, config: SplittingConfig = None):
        self.config = config or SplittingConfig()

        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config.tiktoken_model)
        except KeyError:
            # Fallback to a known encoding if model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_content(
        self, content: str, headers: List[Header], frontmatter: Dict[str, Any] = None
    ) -> List[SectionData]:
        """
        Split markdown content into optimally-sized sections.

        Args:
            content: Markdown content (without frontmatter)
            headers: List of extracted headers
            frontmatter: Optional frontmatter data

        Returns:
            List of SectionData objects ready for chunk assembly
        """
        sections = []

        # Handle frontmatter if present
        if frontmatter:
            frontmatter_content = self._format_frontmatter(frontmatter)
            frontmatter_tokens = self._count_tokens(frontmatter_content)

            sections.append(
                SectionData(
                    content=frontmatter_content,
                    header=None,
                    start_position=0,
                    end_position=0,
                    section_type="frontmatter",
                    token_count=frontmatter_tokens,
                )
            )

        if not headers:
            # Document without headers - treat as single section
            token_count = self._count_tokens(content)
            sections.append(
                SectionData(
                    content=content,
                    header=None,
                    start_position=0,
                    end_position=len(content),
                    section_type="document",
                    token_count=token_count,
                )
            )
            return sections

        # Calculate section boundaries based on headers
        boundaries = self._calculate_section_boundaries(headers, content)

        # Group by top-level sections (H1, or H2 if no H1)
        top_level_sections = self._group_by_top_level(boundaries)

        # Apply adaptive splitting to each top-level section
        for top_section in top_level_sections:
            sections.extend(self._adaptive_split_section(top_section, content))

        return sections

    def _format_frontmatter(self, frontmatter: Dict[str, Any]) -> str:
        """Format frontmatter as YAML string."""
        import yaml

        return f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _calculate_section_boundaries(self, headers: List[Header], content: str) -> List[SectionBoundary]:
        """Calculate content boundaries for each header section."""
        boundaries = []
        content_length = len(content)
        content_lines = content.split("\n")

        # Handle content before first header
        if headers and headers[0].line_number > 1:
            pre_content_lines = content_lines[: headers[0].line_number - 1]
            pre_content = "\n".join(pre_content_lines)
            if pre_content.strip():
                boundaries.append(
                    SectionBoundary(
                        header=None,
                        start_position=0,
                        end_position=headers[0].position,
                        content=pre_content,
                        section_type="document",
                    )
                )

        # Process each header section
        for i, header in enumerate(headers):
            start_pos = header.position

            # Find end position (start of next header at same or higher level)
            end_pos = content_length
            for j in range(i + 1, len(headers)):
                next_header = headers[j]
                if next_header.level <= header.level:
                    end_pos = next_header.position
                    break

            # Extract section content
            start_line = header.line_number - 1
            end_line = len(content_lines)

            if end_pos < content_length:
                # Find line number for end position
                chars_seen = 0
                for line_idx, line in enumerate(content_lines):
                    if chars_seen >= end_pos:
                        end_line = line_idx
                        break
                    chars_seen += len(line) + 1  # +1 for newline

            section_lines = content_lines[start_line:end_line]
            section_content = "\n".join(section_lines)

            boundaries.append(
                SectionBoundary(
                    header=header,
                    start_position=start_pos,
                    end_position=end_pos,
                    content=section_content,
                    section_type="section",
                )
            )

        return boundaries

    def _group_by_top_level(self, boundaries: List[SectionBoundary]) -> List[List[SectionBoundary]]:
        """Group boundaries by top-level sections (H1, or H2 if no H1)."""
        if not boundaries:
            return []

        # Find the top level (lowest header level number)
        header_boundaries = [b for b in boundaries if b.header is not None]
        if not header_boundaries:
            return [boundaries]  # No headers, return all as single group

        top_level = min(b.header.level for b in header_boundaries)

        groups = []
        current_group = []

        for boundary in boundaries:
            if boundary.header is None or boundary.header.level > top_level:
                # Pre-header content or subsection
                current_group.append(boundary)
            else:
                # Top-level header
                if current_group:
                    groups.append(current_group)
                current_group = [boundary]

        if current_group:
            groups.append(current_group)

        return groups

    def _adaptive_split_section(self, section_group: List[SectionBoundary], full_content: str) -> List[SectionData]:
        """Apply adaptive splitting to a top-level section group."""
        if not section_group:
            return []

        # Combine all content in the group
        combined_content = "\n\n".join(boundary.content for boundary in section_group)
        total_tokens = self._count_tokens(combined_content)

        # If within token limit, keep as single section
        if total_tokens <= self.config.max_tokens:
            main_header = next((b.header for b in section_group if b.header), None)
            return [
                SectionData(
                    content=combined_content,
                    header=main_header,
                    start_position=section_group[0].start_position,
                    end_position=section_group[-1].end_position,
                    section_type=section_group[0].section_type,
                    token_count=total_tokens,
                )
            ]

        # Section too large - needs splitting
        return self._split_large_section(section_group, full_content)

    def _split_large_section(self, section_group: List[SectionBoundary], full_content: str) -> List[SectionData]:
        """Split a large section into smaller chunks with context preservation."""
        if len(section_group) == 1:
            # Single boundary that's too large - try to split by paragraphs or keep as is
            boundary = section_group[0]
            token_count = self._count_tokens(boundary.content)

            # If it's still too large, we could implement paragraph-level splitting here
            # For now, we'll keep large single sections as-is (they're unsplittable at header level)
            return [
                SectionData(
                    content=boundary.content,
                    header=boundary.header,
                    start_position=boundary.start_position,
                    end_position=boundary.end_position,
                    section_type=boundary.section_type,
                    token_count=token_count,
                    is_split_section=False,  # Not split because it's a single atomic section
                )
            ]

        # Multiple boundaries - split them
        main_boundary = section_group[0]  # Usually the top-level header
        sub_boundaries = section_group[1:]

        sections = []

        # Create context string from main header
        main_context = ""
        if main_boundary.header:
            main_context = f"{'#' * main_boundary.header.level} {main_boundary.header.text}"

            # Add any content before first sub-boundary
            main_content_lines = main_boundary.content.split("\n")
            if len(main_content_lines) > 1:  # More than just the header
                content_before_sub = []
                for line in main_content_lines[1:]:  # Skip header line
                    if line.strip() and not line.strip().startswith("#"):
                        content_before_sub.append(line)
                    elif line.strip().startswith("#"):
                        break  # Stop at next header

                if content_before_sub:
                    main_context += "\n\n" + "\n".join(content_before_sub)

        # First section: main header + content + first subsection
        if sub_boundaries:
            first_content = main_boundary.content
            if not first_content.endswith("\n\n") and sub_boundaries[0].content:
                first_content += "\n\n"
            first_content += sub_boundaries[0].content

            first_tokens = self._count_tokens(first_content)
            sections.append(
                SectionData(
                    content=first_content,
                    header=main_boundary.header,
                    start_position=main_boundary.start_position,
                    end_position=sub_boundaries[0].end_position,
                    section_type="section",
                    token_count=first_tokens,
                    is_split_section=True,
                    split_index=1,
                    total_splits=(len(sub_boundaries) + 1 if len(sub_boundaries) > 1 else 1),
                    split_parent_id=(self._generate_section_id(main_boundary.header) if main_boundary.header else None),
                )
            )

        # Remaining subsections with context
        for i, boundary in enumerate(sub_boundaries[1:], 2):
            # Add context if configured
            section_content = boundary.content
            if self.config.include_context_in_splits and main_context:
                if boundary.header:
                    context_header = f"{main_context} > {boundary.header.text}"
                    section_content = f"{context_header}\n\n{boundary.content}"
                else:
                    section_content = f"{main_context}\n\n{boundary.content}"

            token_count = self._count_tokens(section_content)
            sections.append(
                SectionData(
                    content=section_content,
                    header=boundary.header,
                    start_position=boundary.start_position,
                    end_position=boundary.end_position,
                    section_type="section",
                    token_count=token_count,
                    is_split_section=True,
                    split_index=i,
                    total_splits=(len(sub_boundaries) + 1 if len(sub_boundaries) > 1 else 1),
                    split_parent_id=(self._generate_section_id(main_boundary.header) if main_boundary.header else None),
                    parent_context=main_context,
                )
            )

        return sections

    def _generate_section_id(self, header: Header) -> str:
        """Generate section ID from header text."""
        if not header:
            return "unknown"

        # Slugify header text
        slug = re.sub(r"[^\w\s-]", "", header.text.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")
