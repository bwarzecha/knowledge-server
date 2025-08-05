"""Integration tests for markdown processor components."""

from pathlib import Path

from src.markdown_processor.chunk_assembler import ChunkAssembler
from src.markdown_processor.header_extractor import HeaderExtractor
from src.markdown_processor.parser import MarkdownParser
from src.markdown_processor.scanner import DirectoryScanner
from src.markdown_processor.section_splitter import (AdaptiveSectionSplitter,
                                                     SplittingConfig)


class TestMarkdownProcessorIntegration:
    """Test the full markdown processing pipeline."""

    def test_full_pipeline_with_sample_file(self):
        """Test the complete pipeline with an actual sample file."""
        # Setup components
        scanner = DirectoryScanner()
        parser = MarkdownParser()
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=1000))
        assembler = ChunkAssembler()

        # Scan for files
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        files = list(scanner.scan_for_markdown_files(str(samples_dir)))
        assert len(files) > 0

        # Process first markdown file
        first_file = files[0]
        file_path = samples_dir / first_file

        # Phase 1: Parse file
        parse_result = parser.parse_file(file_path)
        assert parse_result.success

        # Phase 2: Extract headers
        headers = header_extractor.extract_headers(parse_result.content)

        # Phase 3: Split sections
        sections = splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )
        assert len(sections) > 0

        # Phase 4: Assemble chunks
        chunks = assembler.assemble_chunks(
            sections, parse_result.frontmatter, first_file
        )
        assert len(chunks) == len(sections)

        # Verify chunk structure
        for chunk in chunks:
            assert "id" in chunk
            assert "document" in chunk
            assert "metadata" in chunk

            # Check ID format
            assert ":" in chunk["id"]
            assert chunk["id"].startswith(first_file)

            # Check metadata completeness
            metadata = chunk["metadata"]
            assert "type" in metadata
            assert "source_file" in metadata
            assert "section_level" in metadata
            assert "title" in metadata
            assert "word_count" in metadata
            assert "content_hash" in metadata

            # Check navigation metadata
            assert "previous_chunk" in metadata
            assert "next_chunk" in metadata
            assert "parent_section" in metadata
            assert "child_sections" in metadata

            # Check content analysis
            assert "has_code_blocks" in metadata
            assert "has_tables" in metadata
            assert "cross_doc_refs" in metadata
            assert "external_links" in metadata

    def test_navigation_consistency(self):
        """Test that navigation relationships are consistent."""
        parser = MarkdownParser()
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=1000))
        assembler = ChunkAssembler()

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        parse_result = parser.parse_file(sample_file)
        headers = header_extractor.extract_headers(parse_result.content)
        sections = splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )
        chunks = assembler.assemble_chunks(
            sections, parse_result.frontmatter, "test.md"
        )

        # Build chunk lookup
        chunk_lookup = {chunk["id"]: chunk for chunk in chunks}

        # Verify navigation consistency
        for chunk in chunks:
            metadata = chunk["metadata"]
            chunk_id = chunk["id"]

            # Check previous/next bidirectionality
            if metadata.get("previous_chunk"):
                prev_id = metadata["previous_chunk"]
                assert prev_id in chunk_lookup, f"Previous chunk {prev_id} not found"
                prev_chunk = chunk_lookup[prev_id]
                assert prev_chunk["metadata"].get("next_chunk") == chunk_id

            if metadata.get("next_chunk"):
                next_id = metadata["next_chunk"]
                assert next_id in chunk_lookup, f"Next chunk {next_id} not found"
                next_chunk = chunk_lookup[next_id]
                assert next_chunk["metadata"].get("previous_chunk") == chunk_id

            # Check parent-child relationships
            if metadata.get("parent_section"):
                parent_id = metadata["parent_section"]
                assert parent_id in chunk_lookup, f"Parent {parent_id} not found"
                parent_chunk = chunk_lookup[parent_id]
                assert chunk_id in parent_chunk["metadata"].get("child_sections", [])

    def test_content_preservation(self):
        """Test that content is preserved correctly through the pipeline."""
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=2000))
        assembler = ChunkAssembler()

        # Create test content
        test_content = """# Test Document

This is the introduction.

## Section 1

Content with **bold** and *italic* text.

### Subsection 1.1

Here's a code block:

```python
def hello():
    print("Hello, world!")
```

## Section 2

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

And a [link](https://example.com)."""

        # Process content
        headers = header_extractor.extract_headers(test_content)
        sections = splitter.split_content(test_content, headers, None)
        chunks = assembler.assemble_chunks(sections, None, "test.md")

        # Verify content preservation
        all_content = "\n\n".join(chunk["document"] for chunk in chunks)

        # Check key elements are preserved
        assert "# Test Document" in all_content
        assert "**bold**" in all_content
        assert "*italic*" in all_content
        assert "```python" in all_content
        assert "def hello():" in all_content
        assert "| Column 1 | Column 2 |" in all_content
        assert "[link](https://example.com)" in all_content

    def test_frontmatter_propagation(self):
        """Test that frontmatter data propagates to all chunks."""
        parser = MarkdownParser()
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter()
        assembler = ChunkAssembler()

        # Create content with frontmatter
        test_content = """---
title: Test Document
author: Test Author
tags:
  - test
  - markdown
category: documentation
---

# Main Title

Content here.

## Section 1

More content."""

        # Parse with frontmatter
        parse_result = parser.parse_content(test_content)
        headers = header_extractor.extract_headers(parse_result.content)
        sections = splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )
        chunks = assembler.assemble_chunks(
            sections, parse_result.frontmatter, "test.md"
        )

        # Check frontmatter propagation
        for chunk in chunks:
            metadata = chunk["metadata"]
            if chunk["metadata"]["type"] != "markdown_frontmatter":
                # Document metadata should be present
                assert metadata["document_title"] == "Test Document"
                assert metadata["document_tags"] == ["test", "markdown"]
                assert metadata["document_category"] == "documentation"

    def test_split_section_handling(self):
        """Test handling of split sections with large content."""
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(
            SplittingConfig(max_tokens=50, include_context_in_splits=True)
        )
        assembler = ChunkAssembler()

        # Create content that will force splitting
        test_content = """# Large Document

This is a large document that will be split.

## Section 1

This section has a lot of content that exceeds our token limit.
It includes multiple paragraphs and should be split.

## Section 2

Another section with substantial content that will also need splitting.
This ensures we test the split functionality properly.

## Section 3

Final section with more content to ensure proper splitting behavior."""

        headers = header_extractor.extract_headers(test_content)
        sections = splitter.split_content(test_content, headers, None)
        chunks = assembler.assemble_chunks(sections, None, "test.md")

        # Find split sections
        split_chunks = [
            c for c in chunks if c["metadata"].get("is_split_section", False)
        ]

        if split_chunks:
            # Verify split metadata
            for chunk in split_chunks:
                metadata = chunk["metadata"]
                assert "split_index" in metadata
                assert "total_splits" in metadata
                assert "split_parent_id" in metadata

                # Check context preservation in splits
                if metadata["split_index"] > 1:
                    assert "Large Document" in chunk["document"] or metadata.get(
                        "hierarchical_context"
                    )

    def test_reference_extraction(self):
        """Test that references are extracted correctly."""
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter()
        assembler = ChunkAssembler()

        test_content = """# Document with Links

See the [external site](https://example.com).

Check the [internal section](#section-2).

Also see [another document](other.md).

## Section 2

Content here."""

        headers = header_extractor.extract_headers(test_content)
        sections = splitter.split_content(test_content, headers, None)
        chunks = assembler.assemble_chunks(sections, None, "test.md")

        # Check first chunk (main content)
        main_chunk = chunks[0]
        metadata = main_chunk["metadata"]

        assert len(metadata["external_links"]) > 0
        assert "https://example.com" in metadata["external_links"]
        assert len(metadata["internal_links"]) > 0
        assert "#section-2" in metadata["internal_links"]
        assert len(metadata["cross_doc_refs"]) > 0
        assert "other.md" in metadata["cross_doc_refs"]

    def test_content_analysis_metadata(self):
        """Test content analysis features."""
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter()
        assembler = ChunkAssembler()

        test_content = """# Content Analysis Test

Here's some text with various elements.

## Code Section

```python
def example():
    return True
```

## Table Section

| Header 1 | Header 2 |
|----------|----------|
| Data 1   | Data 2   |

## Image Section

![Alt text](image.png)

And more text here."""

        headers = header_extractor.extract_headers(test_content)
        sections = splitter.split_content(test_content, headers, None)
        chunks = assembler.assemble_chunks(sections, None, "test.md")

        # Find chunks with different content types
        for chunk in chunks:
            metadata = chunk["metadata"]
            content = chunk["document"]

            if "```python" in content:
                assert metadata["has_code_blocks"] is True

            if "| Header 1 |" in content:
                assert metadata["has_tables"] is True

            if "![Alt text]" in content:
                assert metadata["has_images"] is True

            # All chunks should have word count
            assert metadata["word_count"] >= 0

    def test_chunk_validation(self):
        """Test chunk validation functionality."""
        assembler = ChunkAssembler()

        # Create valid chunks
        valid_chunks = [
            {
                "id": "test.md:section-1",
                "document": "Content 1",
                "metadata": {
                    "type": "markdown_section",
                    "source_file": "test.md",
                    "section_level": 1,
                    "title": "Section 1",
                    "word_count": 2,
                    "chunk_index": 0,
                    "total_chunks": 2,
                    "content_hash": "abc123",
                    "next_chunk": "test.md:section-2",
                },
            },
            {
                "id": "test.md:section-2",
                "document": "Content 2",
                "metadata": {
                    "type": "markdown_section",
                    "source_file": "test.md",
                    "section_level": 1,
                    "title": "Section 2",
                    "word_count": 2,
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "content_hash": "def456",
                    "previous_chunk": "test.md:section-1",
                },
            },
        ]

        # Should validate successfully
        errors = assembler.validate_chunks(valid_chunks)
        assert len(errors) == 0

        # Create invalid chunks
        invalid_chunks = [
            {
                "id": "test.md:section-1",
                "document": "Content",
                "metadata": {
                    "type": "markdown_section",
                    # Missing required fields
                },
            }
        ]

        # Should have validation errors
        errors = assembler.validate_chunks(invalid_chunks)
        assert len(errors) > 0
