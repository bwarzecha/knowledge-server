"""Tests for adaptive section splitter."""

from pathlib import Path

from src.markdown_processor.header_extractor import HeaderExtractor
from src.markdown_processor.parser import MarkdownParser
from src.markdown_processor.section_splitter import (AdaptiveSectionSplitter,
                                                     SplittingConfig)


class TestAdaptiveSectionSplitter:
    """Test the AdaptiveSectionSplitter component."""

    def test_split_sample_file_small_token_limit(self):
        """Test splitting sample file with small token limit to force splits."""
        # Setup components
        parser = MarkdownParser()
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=500))

        # Parse sample file
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        parse_result = parser.parse_file(sample_file)
        assert parse_result.success

        # Extract headers
        headers = header_extractor.extract_headers(parse_result.content)

        # Split content
        sections = splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )

        # Should have multiple sections due to small token limit
        assert len(sections) > 3

        # First section should be frontmatter
        assert sections[0].section_type == "frontmatter"
        assert sections[0].is_split_section is False

        # Check that sections either:
        # 1. Are within token limit, OR
        # 2. Are large single sections that can't be split further (no subsections)
        for section in sections[1:]:  # Skip frontmatter
            if section.token_count > 700:
                # Large sections should be atomic (no subsections to split)
                assert (
                    not section.is_split_section
                ), f"Large section should be atomic: {section.header.text if section.header else 'No header'}"
            else:
                # Normal sized sections should be within limit
                assert section.token_count <= 700

    def test_split_sample_file_large_token_limit(self):
        """Test splitting sample file with large token limit to avoid unnecessary splits."""
        parser = MarkdownParser()
        header_extractor = HeaderExtractor()
        splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=2000))

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        parse_result = parser.parse_file(sample_file)
        headers = header_extractor.extract_headers(parse_result.content)

        sections = splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )

        # Should have fewer sections with larger token limit
        assert len(sections) >= 1

        # Each section should be under the token limit
        for section in sections:
            assert section.token_count <= 2000

    def test_split_content_without_headers(self):
        """Test splitting content without headers."""
        splitter = AdaptiveSectionSplitter()

        content = """This is a document without any headers.

It has multiple paragraphs of content that might be long enough to require splitting based on token count.

This is another paragraph with more content to test the behavior."""

        sections = splitter.split_content(content, [], None)

        assert len(sections) == 1
        assert sections[0].section_type == "document"
        assert sections[0].header is None
        assert sections[0].is_split_section is False

    def test_split_with_frontmatter(self):
        """Test splitting content with frontmatter."""
        splitter = AdaptiveSectionSplitter()

        content = """# Introduction

This is the introduction section.

## Getting Started

This is the getting started section."""

        frontmatter = {
            "title": "Test Document",
            "author": "Test Author",
            "tags": ["test", "markdown"],
        }

        headers = [
            type(
                "Header",
                (),
                {"level": 1, "text": "Introduction", "position": 0, "line_number": 1},
            ),
            type(
                "Header",
                (),
                {
                    "level": 2,
                    "text": "Getting Started",
                    "position": 50,
                    "line_number": 5,
                },
            ),
        ]

        sections = splitter.split_content(content, headers, frontmatter)

        # Should have frontmatter + content sections
        assert len(sections) >= 2
        assert sections[0].section_type == "frontmatter"
        assert "title: Test Document" in sections[0].content

    def test_adaptive_splitting_large_section(self):
        """Test adaptive splitting when a section is too large."""
        config = SplittingConfig(max_tokens=100, include_context_in_splits=True)
        splitter = AdaptiveSectionSplitter(config)

        # Create content that will exceed token limit
        large_content = """# Main Section

This is the introduction to the main section with quite a bit of content.

## Subsection 1

This is subsection 1 with substantial content that should make the overall section exceed the token limit when
combined with other subsections.

## Subsection 2

This is subsection 2 with even more substantial content that will definitely push us over the token limit for
the main section.

## Subsection 3

And this is subsection 3 with additional content that makes the splitting necessary."""

        headers = [
            type(
                "Header",
                (),
                {"level": 1, "text": "Main Section", "position": 0, "line_number": 1},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Subsection 1", "position": 80, "line_number": 5},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Subsection 2", "position": 200, "line_number": 9},
            ),
            type(
                "Header",
                (),
                {
                    "level": 2,
                    "text": "Subsection 3",
                    "position": 350,
                    "line_number": 13,
                },
            ),
        ]

        sections = splitter.split_content(large_content, headers, None)

        # Should be split into multiple sections
        assert len(sections) > 1

        # Check for split metadata
        split_sections = [s for s in sections if s.is_split_section]
        if split_sections:
            assert split_sections[0].split_index == 1
            assert all(
                s.split_parent_id == split_sections[0].split_parent_id
                for s in split_sections
            )

    def test_context_preservation_in_splits(self):
        """Test that context is preserved when sections are split."""
        config = SplittingConfig(max_tokens=50, include_context_in_splits=True)
        splitter = AdaptiveSectionSplitter(config)

        content = """# API Guide

Introduction to the API.

## Authentication

How to authenticate.

## Endpoints

Available endpoints."""

        headers = [
            type(
                "Header",
                (),
                {"level": 1, "text": "API Guide", "position": 0, "line_number": 1},
            ),
            type(
                "Header",
                (),
                {
                    "level": 2,
                    "text": "Authentication",
                    "position": 40,
                    "line_number": 5,
                },
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Endpoints", "position": 80, "line_number": 9},
            ),
        ]

        sections = splitter.split_content(content, headers, None)

        # Find split sections with context
        split_sections = [
            s for s in sections if s.is_split_section and s.split_index > 1
        ]

        if split_sections:
            # Should include parent context
            for section in split_sections:
                assert "API Guide" in section.content
                assert section.parent_context is not None

    def test_token_counting_accuracy(self):
        """Test that token counting is working correctly."""
        splitter = AdaptiveSectionSplitter()

        test_content = "This is a test sentence with some words."
        token_count = splitter._count_tokens(test_content)

        # Should be a reasonable number of tokens
        assert token_count > 5
        assert token_count < 20

    def test_section_boundary_calculation(self):
        """Test section boundary calculation."""
        splitter = AdaptiveSectionSplitter()

        content = """# Title

Content after title.

## Section 1

Content for section 1.

## Section 2

Content for section 2."""

        headers = [
            type(
                "Header",
                (),
                {"level": 1, "text": "Title", "position": 0, "line_number": 1},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Section 1", "position": 30, "line_number": 5},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Section 2", "position": 70, "line_number": 9},
            ),
        ]

        boundaries = splitter._calculate_section_boundaries(headers, content)

        assert len(boundaries) == 3
        assert all(b.content for b in boundaries)  # All should have content

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        splitter = AdaptiveSectionSplitter()

        sections = splitter.split_content("", [], None)

        assert len(sections) == 1
        assert sections[0].section_type == "document"
        assert sections[0].content == ""

    def test_configuration_parameters(self):
        """Test different configuration parameters."""
        # Test without context in splits
        config = SplittingConfig(include_context_in_splits=False, max_tokens=50)
        splitter = AdaptiveSectionSplitter(config)

        content = """# Main

## Sub 1

Content 1.

## Sub 2

Content 2."""

        headers = [
            type(
                "Header",
                (),
                {"level": 1, "text": "Main", "position": 0, "line_number": 1},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Sub 1", "position": 10, "line_number": 3},
            ),
            type(
                "Header",
                (),
                {"level": 2, "text": "Sub 2", "position": 30, "line_number": 7},
            ),
        ]

        sections = splitter.split_content(content, headers, None)

        # Should still work without context
        assert len(sections) >= 1
