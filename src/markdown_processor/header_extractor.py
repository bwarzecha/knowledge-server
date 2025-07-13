"""Header extractor for analyzing markdown header structure."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Header:
    """Represents a markdown header with position and hierarchy information."""

    level: int  # 1-6 (number of # symbols)
    text: str  # Header text without # symbols
    position: int  # Character position in content
    line_number: int  # Line number (1-based)
    raw_line: str  # Full line including # symbols


class HeaderExtractor:
    """Extracts header structure from markdown content."""

    def __init__(self):
        # Regex pattern to match markdown headers (1-6 levels)
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def extract_headers(self, content: str) -> List[Header]:
        """
        Extract all headers from markdown content.

        Args:
            content: Markdown content string

        Returns:
            List of Header objects in document order
        """
        headers = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            match = self.header_pattern.match(line.strip())
            if match:
                hash_symbols = match.group(1)
                header_text = match.group(2).strip()
                level = len(hash_symbols)

                # Calculate character position in full content
                position = sum(len(line) + 1 for line in lines[: line_num - 1])

                header = Header(
                    level=level,
                    text=header_text,
                    position=position,
                    line_number=line_num,
                    raw_line=line.strip(),
                )
                headers.append(header)

        return headers

    def build_header_hierarchy(self, headers: List[Header]) -> List[dict]:
        """
        Build hierarchical structure from flat header list.

        Args:
            headers: List of Header objects

        Returns:
            List of header dictionaries with hierarchy information
        """
        if not headers:
            return []

        hierarchical_headers = []
        parent_stack = []  # Stack to track parent headers at each level

        for header in headers:
            # Pop parents that are at same or deeper level
            while parent_stack and parent_stack[-1]["level"] >= header.level:
                parent_stack.pop()

            # Build header data with hierarchy info
            header_data = {
                "level": header.level,
                "text": header.text,
                "position": header.position,
                "line_number": header.line_number,
                "raw_line": header.raw_line,
                "parent": parent_stack[-1] if parent_stack else None,
                "children": [],
                "section_path": self._build_section_path(parent_stack, header.text),
            }

            # Add as child to parent
            if parent_stack:
                parent_stack[-1]["children"].append(header_data)

            # Add to results and parent stack
            hierarchical_headers.append(header_data)
            parent_stack.append(header_data)

        return hierarchical_headers

    def _build_section_path(self, parent_stack: List[dict], current_text: str) -> str:
        """
        Build hierarchical section path like 'intro/getting-started/installation'.

        Args:
            parent_stack: Current parent headers
            current_text: Current header text

        Returns:
            Section path string
        """
        path_parts = []

        # Add parent paths
        for parent in parent_stack:
            path_parts.append(self._slugify(parent["text"]))

        # Add current header
        path_parts.append(self._slugify(current_text))

        return "/".join(path_parts)

    def _slugify(self, text: str) -> str:
        """
        Convert header text to URL-friendly slug.

        Args:
            text: Header text

        Returns:
            Slugified text
        """
        # Convert to lowercase and replace spaces/special chars with dashes
        # Keep periods as dashes, then clean up
        slug = re.sub(r"[^\w\s.-]", "", text.lower())
        slug = re.sub(r"[.\s]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)  # Multiple dashes to single dash
        return slug.strip("-")

    def generate_section_identifier(self, text: str, existing_ids: set) -> str:
        """
        Generate unique section identifier, handling duplicates.

        Args:
            text: Header text
            existing_ids: Set of already used identifiers

        Returns:
            Unique section identifier
        """
        base_id = self._slugify(text)

        if base_id not in existing_ids:
            return base_id

        # Handle duplicates by appending numbers
        counter = 2
        while f"{base_id}-{counter}" in existing_ids:
            counter += 1

        return f"{base_id}-{counter}"

    def get_section_boundaries(self, headers: List[Header], content: str) -> List[dict]:
        """
        Calculate content boundaries for each header section.

        Args:
            headers: List of Header objects
            content: Full markdown content

        Returns:
            List of section boundary information
        """
        if not headers:
            return []

        sections = []
        content_length = len(content)

        for i, header in enumerate(headers):
            start_pos = header.position

            # Find end position (start of next header at same or higher level)
            end_pos = content_length
            for j in range(i + 1, len(headers)):
                next_header = headers[j]
                if next_header.level <= header.level:
                    end_pos = next_header.position
                    break

            sections.append(
                {
                    "header": header,
                    "start_position": start_pos,
                    "end_position": end_pos,
                    "content_lines": self._get_line_range(content, start_pos, end_pos),
                }
            )

        return sections

    def _get_line_range(self, content: str, start_pos: int, end_pos: int) -> tuple:
        """
        Get line number range for content between positions.

        Args:
            content: Full content
            start_pos: Start character position
            end_pos: End character position

        Returns:
            Tuple of (start_line, end_line)
        """
        lines_before_start = content[:start_pos].count("\n")
        lines_before_end = content[:end_pos].count("\n")

        return (lines_before_start + 1, lines_before_end + 1)
