"""Reference scanner for extracting links and references from markdown content."""

import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse


@dataclass
class LinkReference:
    """Represents a found link reference."""

    text: str  # Link text
    url: str  # Link URL/target
    link_type: str  # "internal", "cross_doc", "external", "anchor"
    line_number: int  # Line where link was found
    raw_match: str  # Original markdown text


class ReferenceScanner:
    """Scans markdown content for various types of links and references."""

    def __init__(self):
        # Regex patterns for different link types
        self.markdown_link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
        self.reference_link_pattern = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
        self.autolink_pattern = re.compile(r"<(https?://[^>]+)>")
        self.url_pattern = re.compile(r"https?://[^\s\]\)]+")

        # Header anchor pattern for internal links
        self.anchor_pattern = re.compile(r"#([a-zA-Z0-9-_]+)")

    def scan_references(self, content: str, source_file: str = None) -> dict:
        """
        Scan content for all types of references.

        Args:
            content: Markdown content to scan
            source_file: Optional source file path for relative link resolution

        Returns:
            Dictionary with categorized references
        """
        lines = content.split("\n")

        internal_links = []
        cross_doc_refs = []
        external_links = []
        anchor_links = []

        for line_num, line in enumerate(lines, 1):
            # Find markdown links [text](url)
            for match in self.markdown_link_pattern.finditer(line):
                link_text = match.group(1)
                link_url = match.group(2)
                raw_match = match.group(0)

                link_ref = LinkReference(
                    text=link_text,
                    url=link_url,
                    link_type=self._classify_link(link_url),
                    line_number=line_num,
                    raw_match=raw_match,
                )

                # Categorize based on link type
                if link_ref.link_type == "internal":
                    anchor_links.append(link_ref)
                elif link_ref.link_type == "cross_doc":
                    cross_doc_refs.append(link_ref)
                elif link_ref.link_type == "external":
                    external_links.append(link_ref)

            # Find autolinks <http://example.com>
            for match in self.autolink_pattern.finditer(line):
                link_url = match.group(1)
                raw_match = match.group(0)

                link_ref = LinkReference(
                    text=link_url,  # Autolinks use URL as text
                    url=link_url,
                    link_type="external",
                    line_number=line_num,
                    raw_match=raw_match,
                )
                external_links.append(link_ref)

        return {
            "internal_links": self._deduplicate_links(anchor_links),
            "cross_doc_refs": self._deduplicate_links(cross_doc_refs),
            "external_links": self._deduplicate_links(external_links),
            "all_links": self._deduplicate_links(
                internal_links + cross_doc_refs + external_links + anchor_links
            ),
        }

    def _classify_link(self, url: str) -> str:
        """
        Classify a link URL into type.

        Args:
            url: Link URL to classify

        Returns:
            Link type: "internal", "cross_doc", "external"
        """
        url = url.strip()

        # Internal anchor links (#section)
        if url.startswith("#"):
            return "internal"

        # Check if it's a full URL
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            return "external"

        # Relative links to other markdown files
        if url.endswith(".md") or url.endswith(".markdown"):
            return "cross_doc"

        # Links to markdown files with anchors
        if ".md#" in url or ".markdown#" in url:
            return "cross_doc"

        # Other relative links (assume cross-doc)
        if not parsed.scheme and ("/" in url or url.endswith(".html")):
            return "cross_doc"

        # Default to external
        return "external"

    def _deduplicate_links(self, links: List[LinkReference]) -> List[str]:
        """
        Remove duplicate links and return just the URLs.

        Args:
            links: List of LinkReference objects

        Returns:
            List of unique URLs
        """
        seen_urls = set()
        unique_urls = []

        for link in links:
            if link.url not in seen_urls:
                seen_urls.add(link.url)
                unique_urls.append(link.url)

        return unique_urls

    def extract_cross_document_references(
        self, content: str, source_file: str = None
    ) -> List[str]:
        """
        Extract cross-document references and convert to chunk IDs.

        Args:
            content: Markdown content
            source_file: Source file path for relative resolution

        Returns:
            List of potential chunk IDs
        """
        refs = self.scan_references(content, source_file)
        cross_doc_refs = refs["cross_doc_refs"]

        chunk_ids = []
        for ref in cross_doc_refs:
            chunk_id = self._convert_to_chunk_id(ref, source_file)
            if chunk_id:
                chunk_ids.append(chunk_id)

        return chunk_ids

    def _convert_to_chunk_id(self, ref: str, source_file: str = None) -> str:
        """
        Convert a cross-document reference to a chunk ID.

        Args:
            ref: Reference URL (e.g., "../other.md#section")
            source_file: Source file for relative path resolution

        Returns:
            Chunk ID string or None if conversion fails
        """
        try:
            # Handle anchor links in other documents
            if "#" in ref:
                file_part, anchor_part = ref.split("#", 1)
                section_id = self._slugify_anchor(anchor_part)

                # Resolve relative path
                if source_file and file_part:
                    resolved_file = self._resolve_relative_path(file_part, source_file)
                    return f"{resolved_file}:{section_id}"
                elif file_part:
                    return f"{file_part}:{section_id}"
                else:
                    # Just an anchor (#section)
                    return f"{source_file}:{section_id}" if source_file else section_id

            # Handle direct file references
            elif ref.endswith(".md") or ref.endswith(".markdown"):
                if source_file:
                    resolved_file = self._resolve_relative_path(ref, source_file)
                    return f"{resolved_file}:document"
                else:
                    return f"{ref}:document"

        except Exception:
            # If anything goes wrong, return None
            pass

        return None

    def _resolve_relative_path(self, ref_path: str, source_file: str) -> str:
        """
        Resolve relative path reference.

        Args:
            ref_path: Reference path (e.g., "../other.md")
            source_file: Source file path

        Returns:
            Resolved file path
        """
        from pathlib import Path

        try:
            source_dir = Path(source_file).parent
            resolved = (source_dir / ref_path).resolve()
            return str(resolved)
        except Exception:
            # If resolution fails, return original
            return ref_path

    def _slugify_anchor(self, anchor: str) -> str:
        """
        Convert anchor text to section identifier.

        Args:
            anchor: Anchor text

        Returns:
            Slugified section identifier
        """
        # Convert to lowercase and replace special chars
        slug = re.sub(r"[^\w\s-]", "", anchor.lower())
        slug = re.sub(r"[-\s_]+", "-", slug)  # Include underscores in replacement
        return slug.strip("-")

    def find_image_references(self, content: str) -> List[str]:
        """
        Find image references in content.

        Args:
            content: Markdown content

        Returns:
            List of image URLs/paths
        """
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        images = []

        for match in image_pattern.finditer(content):
            image_url = match.group(2)
            images.append(image_url)

        return list(set(images))  # Remove duplicates

    def has_code_blocks(self, content: str) -> bool:
        """
        Check if content contains code blocks.

        Args:
            content: Markdown content

        Returns:
            True if code blocks found
        """
        # Fenced code blocks
        fenced_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        if fenced_pattern.search(content):
            return True

        # Inline code
        inline_pattern = re.compile(r"`[^`]+`")
        if inline_pattern.search(content):
            return True

        # Indented code blocks (4+ spaces)
        lines = content.split("\n")
        for line in lines:
            if line.startswith("    ") and line.strip():
                return True

        return False

    def has_tables(self, content: str) -> bool:
        """
        Check if content contains markdown tables.

        Args:
            content: Markdown content

        Returns:
            True if tables found
        """
        # Look for table separator lines (| --- | --- |)
        table_pattern = re.compile(r"\|[\s]*:?-+:?[\s]*\|", re.MULTILINE)
        return bool(table_pattern.search(content))

    def count_words(self, content: str) -> int:
        """
        Count words in content, excluding markdown syntax.

        Args:
            content: Markdown content

        Returns:
            Word count
        """
        # Remove markdown syntax for more accurate word count
        text = content

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove links but keep link text
        text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

        # Remove headers markdown
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        # Remove emphasis markers
        text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)

        # Split and count words
        words = text.split()
        return len(words)
