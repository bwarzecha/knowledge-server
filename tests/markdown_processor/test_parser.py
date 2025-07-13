"""Tests for markdown parser."""

import tempfile
from pathlib import Path

from src.markdown_processor.parser import MarkdownParser


class TestMarkdownParser:
    """Test the MarkdownParser component."""

    def test_parse_sample_file_with_frontmatter(self):
        """Test parsing actual sample file with frontmatter."""
        parser = MarkdownParser()
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        result = parser.parse_file(sample_file)

        assert result.success is True
        assert result.has_frontmatter is True
        assert result.frontmatter is not None
        assert result.content is not None

        # Check frontmatter content
        assert result.frontmatter["title"] == "Amazon Advertising Advanced Tools Center"
        assert "clippings" in result.frontmatter["tags"]
        # YAML parser converts date strings to date objects
        assert str(result.frontmatter["created"]) == "2025-07-13"

        # Check content starts correctly
        assert result.content.startswith("**Ad groups** are used to group ads")

    def test_parse_markdown_without_frontmatter(self):
        """Test parsing markdown without frontmatter."""
        parser = MarkdownParser()

        content = """# Simple Markdown

This is just regular markdown content without frontmatter.

## Section

Some content here."""

        result = parser.parse_content(content)

        assert result.success is True
        assert result.has_frontmatter is False
        assert result.frontmatter == {}
        assert result.content == content

    def test_parse_markdown_with_frontmatter(self):
        """Test parsing markdown with frontmatter."""
        parser = MarkdownParser()

        content = """---
title: Test Document
author: Test Author
tags:
  - test
  - markdown
published: 2024-01-01
---

# Test Document

This is the main content of the document.

## Section 1

Some content here."""

        result = parser.parse_content(content)

        assert result.success is True
        assert result.has_frontmatter is True
        assert result.frontmatter["title"] == "Test Document"
        assert result.frontmatter["author"] == "Test Author"
        assert result.frontmatter["tags"] == ["test", "markdown"]
        # YAML parser converts date strings to date objects
        assert str(result.frontmatter["published"]) == "2024-01-01"
        assert result.content.startswith("# Test Document")

    def test_parse_empty_frontmatter(self):
        """Test parsing markdown with empty frontmatter."""
        parser = MarkdownParser()

        content = """---
---

# Document

Content without frontmatter data."""

        result = parser.parse_content(content)

        assert result.success is True
        assert result.has_frontmatter is False  # Empty frontmatter treated as no frontmatter
        assert result.frontmatter == {}
        assert result.content.startswith("# Document")

    def test_parse_invalid_yaml_frontmatter(self):
        """Test error handling for invalid YAML frontmatter."""
        parser = MarkdownParser()

        content = """---
title: Test Document
invalid: [unclosed bracket
---

# Content"""

        result = parser.parse_content(content)

        assert result.success is False
        assert "Invalid YAML frontmatter" in result.error

    def test_parse_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        parser = MarkdownParser()

        result = parser.parse_file("/nonexistent/file.md")

        assert result.success is False
        assert "File not found" in result.error

    def test_parse_directory_instead_of_file(self):
        """Test error handling when path is directory."""
        parser = MarkdownParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = parser.parse_file(temp_dir)

            assert result.success is False
            assert "Path is not a file" in result.error

    def test_parse_unicode_content(self):
        """Test parsing markdown with unicode content."""
        parser = MarkdownParser()

        content = """---
title: Unicode Test
author: José María
---

# Unicode Content

This document contains unicode: 中文, العربية, русский, 🚀"""

        result = parser.parse_content(content)

        assert result.success is True
        assert result.frontmatter["author"] == "José María"
        assert "中文" in result.content
        assert "🚀" in result.content

    def test_parse_complex_frontmatter(self):
        """Test parsing complex frontmatter structures."""
        parser = MarkdownParser()

        content = """---
title: Complex Document
metadata:
  version: 1.0
  authors:
    - name: John Doe
      email: john@example.com
    - name: Jane Smith
      email: jane@example.com
  settings:
    published: true
    draft: false
tags:
  - documentation
  - api
  - guide
---

# Complex Document

Content here."""

        result = parser.parse_content(content)

        assert result.success is True
        assert result.frontmatter["title"] == "Complex Document"
        assert result.frontmatter["metadata"]["version"] == 1.0
        assert len(result.frontmatter["metadata"]["authors"]) == 2
        assert result.frontmatter["metadata"]["authors"][0]["name"] == "John Doe"
        assert result.frontmatter["metadata"]["settings"]["published"] is True
        assert "documentation" in result.frontmatter["tags"]
