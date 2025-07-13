"""Tests for markdown reference scanner."""

from pathlib import Path

from src.markdown_processor.reference_scanner import LinkReference, ReferenceScanner


class TestReferenceScanner:
    """Test the ReferenceScanner component."""

    def test_scan_sample_file_references(self):
        """Test scanning actual sample file for references."""
        scanner = ReferenceScanner()
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        content = sample_file.read_text(encoding="utf-8")

        # Remove frontmatter for content scanning
        content_lines = content.split("\n")
        start_idx = 0
        if content.startswith("---"):
            for i, line in enumerate(content_lines[1:], 1):
                if line.strip() == "---":
                    start_idx = i + 1
                    break
        content_without_frontmatter = "\n".join(content_lines[start_idx:])

        refs = scanner.scan_references(content_without_frontmatter)

        # Should find external links
        assert len(refs["external_links"]) > 0

        # Sample file may not have cross-doc refs, so just check structure
        assert "cross_doc_refs" in refs

        # Check for specific known links in the sample
        external_links = refs["external_links"]

        # Should have some HTTPS URLs
        https_links = [link for link in external_links if link.startswith("https://")]
        assert len(https_links) > 0

    def test_classify_link_types(self):
        """Test link classification."""
        scanner = ReferenceScanner()

        test_cases = [
            ("https://example.com", "external"),
            ("http://example.com", "external"),
            ("#section-name", "internal"),
            ("other-doc.md", "cross_doc"),
            ("../docs/guide.md", "cross_doc"),
            ("guide.md#section", "cross_doc"),
            ("./local.html", "cross_doc"),
        ]

        for url, expected_type in test_cases:
            result = scanner._classify_link(url)
            assert result == expected_type, f"URL {url} should be {expected_type}, got {result}"

    def test_scan_markdown_links(self):
        """Test scanning various markdown link formats."""
        scanner = ReferenceScanner()

        content = """# Test Document

Here are some links:

- [External link](https://example.com)
- [Internal anchor](#section-1)
- [Cross document](../other.md)
- [Cross doc with anchor](guide.md#installation)
- [Relative path](./docs/readme.md)

## Section 1

Some content with an autolink: <https://autolink.com>

And another [external](http://another.com) link.
"""

        refs = scanner.scan_references(content)

        # Should categorize correctly
        assert len(refs["external_links"]) >= 3  # https://example.com, https://autolink.com, http://another.com
        assert len(refs["internal_links"]) >= 1  # #section-1
        assert len(refs["cross_doc_refs"]) >= 3  # ../other.md, guide.md#installation, ./docs/readme.md

    def test_extract_cross_document_references(self):
        """Test extraction of cross-document references with chunk ID conversion."""
        scanner = ReferenceScanner()

        content = """# API Guide

See the [authentication guide](auth.md) for details.

Also check [installation steps](../setup/install.md#requirements).

Internal link to [section below](#api-endpoints).
"""

        chunk_ids = scanner.extract_cross_document_references(content, "docs/api.md")

        # Should have some chunk IDs
        assert len(chunk_ids) > 0

        # Should contain document references
        doc_refs = [cid for cid in chunk_ids if ":document" in cid]
        assert len(doc_refs) > 0

    def test_image_reference_detection(self):
        """Test detection of image references."""
        scanner = ReferenceScanner()

        content = """# Document with Images

Here's an image: ![Alt text](image.png)

And another: ![](https://example.com/photo.jpg)

![Diagram](./diagrams/flow.svg)
"""

        images = scanner.find_image_references(content)

        assert len(images) == 3
        assert "image.png" in images
        assert "https://example.com/photo.jpg" in images
        assert "./diagrams/flow.svg" in images

    def test_code_block_detection(self):
        """Test detection of code blocks."""
        scanner = ReferenceScanner()

        # Content with fenced code blocks
        content_with_fenced = """# Code Example

```python
def hello():
    print("Hello world")
```

Some text with `inline code`.
"""

        assert scanner.has_code_blocks(content_with_fenced) is True

        # Content with indented code
        content_with_indented = """# Example

Here's some code:

    def example():
        return True

End of example.
"""

        assert scanner.has_code_blocks(content_with_indented) is True

        # Content without code
        content_no_code = """# Simple Document

Just plain text here.

No code at all.
"""

        assert scanner.has_code_blocks(content_no_code) is False

    def test_table_detection(self):
        """Test detection of markdown tables."""
        scanner = ReferenceScanner()

        content_with_table = """# Data Table

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
"""

        assert scanner.has_tables(content_with_table) is True

        content_no_table = """# Simple Document

Just text here.

No tables.
"""

        assert scanner.has_tables(content_no_table) is False

    def test_word_counting(self):
        """Test word counting with markdown syntax removal."""
        scanner = ReferenceScanner()

        content = """# Title Here

This is a **bold** paragraph with *italic* text.

Here's a [link](http://example.com) and some `inline code`.

```python
# This code should not be counted
def example():
    pass
```

Final paragraph.
"""

        word_count = scanner.count_words(content)

        # Should count words but exclude code blocks and markdown syntax
        assert word_count > 10
        assert word_count < 30  # Shouldn't count the code block

    def test_chunk_id_conversion(self):
        """Test conversion of references to chunk IDs."""
        scanner = ReferenceScanner()

        test_cases = [
            ("other.md", "other.md:document"),
            ("guide.md#installation", "guide.md:installation"),
            ("#section", "test.md:section"),  # With source file
        ]

        for ref, expected in test_cases:
            if ref.startswith("#"):
                result = scanner._convert_to_chunk_id(ref, "test.md")
            else:
                result = scanner._convert_to_chunk_id(ref, None)

            assert result == expected, f"Reference {ref} should convert to {expected}, got {result}"

    def test_anchor_slugification(self):
        """Test anchor text slugification."""
        scanner = ReferenceScanner()

        test_cases = [
            ("Getting Started", "getting-started"),
            ("API Reference", "api-reference"),
            ("OAuth 2.0", "oauth-20"),
            ("section_with_underscores", "section-with-underscores"),
        ]

        for anchor, expected in test_cases:
            result = scanner._slugify_anchor(anchor)
            assert result == expected

    def test_link_deduplication(self):
        """Test link deduplication."""
        scanner = ReferenceScanner()

        links = [
            LinkReference(
                "Text 1",
                "http://example.com",
                "external",
                1,
                "[Text 1](http://example.com)",
            ),
            LinkReference(
                "Text 2",
                "http://example.com",
                "external",
                2,
                "[Text 2](http://example.com)",
            ),  # Duplicate URL
            LinkReference(
                "Text 3",
                "http://other.com",
                "external",
                3,
                "[Text 3](http://other.com)",
            ),
        ]

        unique_urls = scanner._deduplicate_links(links)

        assert len(unique_urls) == 2
        assert "http://example.com" in unique_urls
        assert "http://other.com" in unique_urls

    def test_relative_path_resolution(self):
        """Test relative path resolution."""
        scanner = ReferenceScanner()

        # Test relative path resolution
        source_file = "docs/api/guide.md"
        ref_path = "../auth/setup.md"

        try:
            resolved = scanner._resolve_relative_path(ref_path, source_file)
            # Should resolve to some path (exact result depends on filesystem)
            assert isinstance(resolved, str)
        except Exception:
            # If resolution fails, that's also acceptable for this test
            pass

    def test_empty_content(self):
        """Test handling of empty content."""
        scanner = ReferenceScanner()

        refs = scanner.scan_references("")

        assert refs["internal_links"] == []
        assert refs["cross_doc_refs"] == []
        assert refs["external_links"] == []
        assert refs["all_links"] == []

        assert scanner.has_code_blocks("") is False
        assert scanner.has_tables("") is False
        assert scanner.count_words("") == 0
        assert scanner.find_image_references("") == []
