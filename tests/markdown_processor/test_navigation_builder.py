"""Tests for NavigationBuilder functionality."""

from src.markdown_processor.header_extractor import Header
from src.markdown_processor.navigation_builder import NavigationBuilder
from src.markdown_processor.section_splitter import SectionData


class TestNavigationBuilder:
    def test_generate_unique_chunk_ids_for_duplicate_headers(self):
        """Test that duplicate headers generate unique chunk IDs."""
        builder = NavigationBuilder()

        # Create sections with duplicate header text "Installation"
        sections = [
            SectionData(
                content="# Installation\n\nFirst installation section",
                header=Header(
                    level=1,
                    text="Installation",
                    position=0,
                    line_number=1,
                    raw_line="# Installation",
                ),
                start_position=0,
                end_position=50,
                section_type="section",
                token_count=10,
            ),
            SectionData(
                content="## Configuration\n\nConfiguration details",
                header=Header(
                    level=2,
                    text="Configuration",
                    position=51,
                    line_number=5,
                    raw_line="## Configuration",
                ),
                start_position=51,
                end_position=100,
                section_type="section",
                token_count=10,
            ),
            SectionData(
                content="# Installation\n\nSecond installation section",
                header=Header(
                    level=1,
                    text="Installation",
                    position=101,
                    line_number=10,
                    raw_line="# Installation",
                ),
                start_position=101,
                end_position=150,
                section_type="section",
                token_count=10,
            ),
            SectionData(
                content="## Installation\n\nThird installation section",
                header=Header(
                    level=2,
                    text="Installation",
                    position=151,
                    line_number=15,
                    raw_line="## Installation",
                ),
                start_position=151,
                end_position=200,
                section_type="section",
                token_count=10,
            ),
        ]

        # Build navigation
        navigation = builder.build_navigation(sections, "test.md")

        # Extract all chunk IDs
        chunk_ids = list(navigation.keys())

        # All chunk IDs should be unique
        assert len(chunk_ids) == len(
            set(chunk_ids)
        ), f"Duplicate chunk IDs found: {chunk_ids}"

        # Verify we have 4 sections
        assert len(chunk_ids) == 4

        # The IDs should be distinguishable even for duplicate headers
        installation_ids = [id for id in chunk_ids if "installation" in id]
        assert len(installation_ids) == 3, "Should have 3 installation sections"
        assert len(set(installation_ids)) == 3, "Installation IDs should all be unique"
