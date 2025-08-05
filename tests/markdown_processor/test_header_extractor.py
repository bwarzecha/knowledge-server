"""Tests for markdown header extractor."""

from pathlib import Path

from src.markdown_processor.header_extractor import HeaderExtractor


class TestHeaderExtractor:
    """Test the HeaderExtractor component."""

    def test_extract_headers_from_sample_file(self):
        """Test extracting headers from actual sample file."""
        extractor = HeaderExtractor()
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center 1.md"

        content = sample_file.read_text(encoding="utf-8")
        # Remove frontmatter for header extraction
        content_lines = content.split("\n")
        start_idx = 0
        if content.startswith("---"):
            # Find end of frontmatter
            for i, line in enumerate(content_lines[1:], 1):
                if line.strip() == "---":
                    start_idx = i + 1
                    break
        content_without_frontmatter = "\n".join(content_lines[start_idx:])

        headers = extractor.extract_headers(content_without_frontmatter)

        assert len(headers) > 0
        # Should find headers like "## Target", "## Targeting types", etc.
        header_texts = [h.text for h in headers]
        assert "Target" in header_texts
        assert "Targeting types" in header_texts

        # Check header levels
        for header in headers:
            assert 1 <= header.level <= 6
            assert header.text
            assert header.position >= 0
            assert header.line_number > 0

    def test_extract_headers_basic(self):
        """Test basic header extraction."""
        extractor = HeaderExtractor()

        content = """# Main Title

Some content here.

## Section 1

Content for section 1.

### Subsection 1.1

More content.

## Section 2

Content for section 2."""

        headers = extractor.extract_headers(content)

        assert len(headers) == 4

        # Check first header
        assert headers[0].level == 1
        assert headers[0].text == "Main Title"
        assert headers[0].line_number == 1

        # Check second header
        assert headers[1].level == 2
        assert headers[1].text == "Section 1"

        # Check nested header
        assert headers[2].level == 3
        assert headers[2].text == "Subsection 1.1"

        # Check last header
        assert headers[3].level == 2
        assert headers[3].text == "Section 2"

    def test_build_header_hierarchy(self):
        """Test building hierarchical structure."""
        extractor = HeaderExtractor()

        content = """# Main Title

## Section 1

### Subsection 1.1

### Subsection 1.2

## Section 2

### Subsection 2.1"""

        headers = extractor.extract_headers(content)
        hierarchical = extractor.build_header_hierarchy(headers)

        assert len(hierarchical) == 6

        # Check main title (no parent)
        main_title = hierarchical[0]
        assert main_title["level"] == 1
        assert main_title["text"] == "Main Title"
        assert main_title["parent"] is None
        assert len(main_title["children"]) == 2  # Should have 2 section children

        # Check section 1
        section1 = hierarchical[1]
        assert section1["level"] == 2
        assert section1["text"] == "Section 1"
        assert section1["parent"] == main_title

        # Check subsection 1.1
        subsection11 = hierarchical[2]
        assert subsection11["level"] == 3
        assert subsection11["text"] == "Subsection 1.1"
        assert subsection11["parent"] == section1

    def test_section_path_generation(self):
        """Test section path generation."""
        extractor = HeaderExtractor()

        content = """# Getting Started

## Installation

### Requirements

## Configuration"""

        headers = extractor.extract_headers(content)
        hierarchical = extractor.build_header_hierarchy(headers)

        # Check section paths
        assert hierarchical[0]["section_path"] == "getting-started"
        assert hierarchical[1]["section_path"] == "getting-started/installation"
        assert (
            hierarchical[2]["section_path"]
            == "getting-started/installation/requirements"
        )
        assert hierarchical[3]["section_path"] == "getting-started/configuration"

    def test_slugify_function(self):
        """Test text slugification."""
        extractor = HeaderExtractor()

        test_cases = [
            ("Getting Started", "getting-started"),
            ("API Reference", "api-reference"),
            ("OAuth 2.0", "oauth-2-0"),
            ("Special Characters! @#$%", "special-characters"),
            ("Multiple   Spaces", "multiple-spaces"),
            ("Trailing Spaces   ", "trailing-spaces"),
            ("   Leading Spaces", "leading-spaces"),
        ]

        for input_text, expected in test_cases:
            result = extractor._slugify(input_text)
            assert result == expected

    def test_generate_section_identifier_with_duplicates(self):
        """Test section identifier generation with duplicates."""
        extractor = HeaderExtractor()
        existing_ids = set()

        # First occurrence
        id1 = extractor.generate_section_identifier("Installation", existing_ids)
        assert id1 == "installation"
        existing_ids.add(id1)

        # Second occurrence (should get -2)
        id2 = extractor.generate_section_identifier("Installation", existing_ids)
        assert id2 == "installation-2"
        existing_ids.add(id2)

        # Third occurrence (should get -3)
        id3 = extractor.generate_section_identifier("Installation", existing_ids)
        assert id3 == "installation-3"

    def test_get_section_boundaries(self):
        """Test section boundary calculation."""
        extractor = HeaderExtractor()

        content = """# Title

Content after title.

## Section 1

Content for section 1.

## Section 2

Content for section 2."""

        headers = extractor.extract_headers(content)
        sections = extractor.get_section_boundaries(headers, content)

        assert len(sections) == 3

        # Check that each section has required fields
        for section in sections:
            assert "header" in section
            assert "start_position" in section
            assert "end_position" in section
            assert "content_lines" in section
            assert section["start_position"] < section["end_position"]

    def test_extract_headers_with_special_formatting(self):
        """Test header extraction with various formatting."""
        extractor = HeaderExtractor()

        content = """## Header with **bold** text

### Header with `code` in it

#### Header with [link](url)

##### Header with multiple    spaces

###### Level 6 header"""

        headers = extractor.extract_headers(content)

        assert len(headers) == 5
        assert headers[0].text == "Header with **bold** text"
        assert headers[1].text == "Header with `code` in it"
        assert headers[2].text == "Header with [link](url)"
        assert headers[3].text == "Header with multiple    spaces"
        assert headers[4].text == "Level 6 header"
        assert headers[4].level == 6

    def test_empty_content(self):
        """Test with empty content."""
        extractor = HeaderExtractor()

        headers = extractor.extract_headers("")
        hierarchical = extractor.build_header_hierarchy(headers)
        sections = extractor.get_section_boundaries(headers, "")

        assert headers == []
        assert hierarchical == []
        assert sections == []

    def test_content_without_headers(self):
        """Test content without any headers."""
        extractor = HeaderExtractor()

        content = """This is just regular content.

No headers here.

Just paragraphs and text."""

        headers = extractor.extract_headers(content)

        assert headers == []
