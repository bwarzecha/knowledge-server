"""Tests for markdown directory scanner."""

import tempfile
from pathlib import Path

import pytest

from src.markdown_processor.scanner import DirectoryScanner, ScannerConfig


class TestDirectoryScanner:
    """Test the DirectoryScanner component."""

    def test_scan_samples_directory(self):
        """Test scanning the actual samples directory."""
        scanner = DirectoryScanner()
        samples_dir = Path(__file__).parent.parent.parent / "samples"

        files = list(scanner.scan_for_markdown_files(str(samples_dir)))

        # Should find the Amazon markdown files
        expected_files = [
            "Amazon Advertising Advanced Tools Center.md",
            "Amazon Advertising Advanced Tools Center 1.md",
            "Amazon Advertising Advanced Tools Center 2.md",
        ]

        assert len(files) == 3
        for expected_file in expected_files:
            assert expected_file in files

    def test_scanner_config_defaults(self):
        """Test scanner configuration defaults."""
        config = ScannerConfig()

        assert config.skip_hidden_files is True
        assert config.supported_extensions == [".md", ".markdown"]
        assert config.process_readme_files is True

    def test_scan_with_custom_extensions(self):
        """Test scanning with custom extensions."""
        config = ScannerConfig(supported_extensions=[".md"])
        scanner = DirectoryScanner(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test.md").write_text("# Test")
            (temp_path / "test.markdown").write_text("# Test")
            (temp_path / "test.txt").write_text("# Test")

            files = list(scanner.scan_for_markdown_files(temp_dir))

            assert len(files) == 1
            assert "test.md" in files
            assert "test.markdown" not in files
            assert "test.txt" not in files

    def test_skip_hidden_files(self):
        """Test skipping hidden files and directories."""
        scanner = DirectoryScanner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create visible and hidden files
            (temp_path / "visible.md").write_text("# Visible")
            (temp_path / ".hidden.md").write_text("# Hidden")

            # Create hidden directory with file
            hidden_dir = temp_path / ".hidden_dir"
            hidden_dir.mkdir()
            (hidden_dir / "nested.md").write_text("# Nested")

            files = list(scanner.scan_for_markdown_files(temp_dir))

            assert len(files) == 1
            assert "visible.md" in files
            assert ".hidden.md" not in files
            assert ".hidden_dir/nested.md" not in files

    def test_recursive_scanning(self):
        """Test recursive directory scanning."""
        scanner = DirectoryScanner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directory structure
            (temp_path / "root.md").write_text("# Root")

            docs_dir = temp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "guide.md").write_text("# Guide")

            nested_dir = docs_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "deep.md").write_text("# Deep")

            files = list(scanner.scan_for_markdown_files(temp_dir))

            assert len(files) == 3
            assert "root.md" in files
            assert "docs/guide.md" in files
            assert "docs/nested/deep.md" in files

    def test_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        scanner = DirectoryScanner()

        with pytest.raises(FileNotFoundError):
            list(scanner.scan_for_markdown_files("/nonexistent/path"))

    def test_file_instead_of_directory(self):
        """Test error handling when path is a file instead of directory."""
        scanner = DirectoryScanner()

        with tempfile.NamedTemporaryFile(suffix=".md") as temp_file:
            with pytest.raises(NotADirectoryError):
                list(scanner.scan_for_markdown_files(temp_file.name))
