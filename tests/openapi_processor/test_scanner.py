"""Tests for Directory Scanner."""

import tempfile
from pathlib import Path

import pytest

from src.openapi_processor.scanner import DirectoryScanner, ScannerConfig


class TestDirectoryScanner:
    """Test cases for DirectoryScanner."""

    def test_scan_empty_directory(self):
        """Test scanning empty directory returns no files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = DirectoryScanner()
            files = list(scanner.scan_for_openapi_files(temp_dir))
            assert files == []

    def test_scan_directory_with_openapi_files(self):
        """Test scanning directory with various OpenAPI files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                "api.json",
                "spec.yaml",
                "openapi.yml",
                "README.md",  # Should be ignored
                "script.py",  # Should be ignored
            ]

            for filename in test_files:
                (Path(temp_dir) / filename).touch()

            scanner = DirectoryScanner()
            files = sorted(list(scanner.scan_for_openapi_files(temp_dir)))

            expected = ["api.json", "openapi.yml", "spec.yaml"]
            assert files == expected

    def test_scan_nested_directories(self):
        """Test scanning nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            sub_dir = Path(temp_dir) / "apis" / "v1"
            sub_dir.mkdir(parents=True)

            # Create files at different levels
            (Path(temp_dir) / "root.json").touch()
            (sub_dir / "nested.yaml").touch()
            (sub_dir / "another.yml").touch()

            scanner = DirectoryScanner()
            files = sorted(list(scanner.scan_for_openapi_files(temp_dir)))

            expected = ["apis/v1/another.yml", "apis/v1/nested.yaml", "root.json"]
            assert files == expected

    def test_skip_hidden_files(self):
        """Test that hidden files are skipped by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create hidden and regular files
            (Path(temp_dir) / ".hidden.json").touch()
            (Path(temp_dir) / "visible.json").touch()

            # Create hidden directory with file
            hidden_dir = Path(temp_dir) / ".hidden_dir"
            hidden_dir.mkdir()
            (hidden_dir / "file.yaml").touch()

            scanner = DirectoryScanner()
            files = list(scanner.scan_for_openapi_files(temp_dir))

            assert files == ["visible.json"]

    def test_include_hidden_files_when_configured(self):
        """Test including hidden files when configured."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / ".hidden.json").touch()
            (Path(temp_dir) / "visible.json").touch()

            config = ScannerConfig(skip_hidden_files=False)
            scanner = DirectoryScanner(config)
            files = sorted(list(scanner.scan_for_openapi_files(temp_dir)))

            assert files == [".hidden.json", "visible.json"]

    def test_custom_extensions(self):
        """Test custom file extensions configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different extensions
            test_files = ["api.json", "spec.yaml", "custom.txt", "other.xml"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()

            config = ScannerConfig(supported_extensions=[".json", ".txt"])
            scanner = DirectoryScanner(config)
            files = sorted(list(scanner.scan_for_openapi_files(temp_dir)))

            assert files == ["api.json", "custom.txt"]

    def test_case_insensitive_extensions(self):
        """Test that file extensions are case insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different case extensions
            test_files = ["api.JSON", "spec.YAML", "other.YML"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()

            scanner = DirectoryScanner()
            files = sorted(list(scanner.scan_for_openapi_files(temp_dir)))

            assert files == ["api.JSON", "other.YML", "spec.YAML"]

    def test_nonexistent_directory_raises_error(self):
        """Test that scanning nonexistent directory raises FileNotFoundError."""
        scanner = DirectoryScanner()
        with pytest.raises(FileNotFoundError):
            list(scanner.scan_for_openapi_files("/path/that/does/not/exist"))

    def test_file_path_instead_of_directory_raises_error(self):
        """Test that passing file path instead of directory raises error."""
        with tempfile.NamedTemporaryFile() as temp_file:
            scanner = DirectoryScanner()
            with pytest.raises(NotADirectoryError):
                list(scanner.scan_for_openapi_files(temp_file.name))
