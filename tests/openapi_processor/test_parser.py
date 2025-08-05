"""Tests for OpenAPI Parser."""

import json
import tempfile
from pathlib import Path

from src.openapi_processor.parser import OpenAPIParser


class TestOpenAPIParser:
    """Test cases for OpenAPIParser."""

    def test_parse_valid_json_file(self):
        """Test parsing a valid JSON OpenAPI file."""
        test_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            parser = OpenAPIParser()
            result = parser.parse_file(temp_file)

            assert result.success is True
            assert result.data == test_data
            assert result.file_type == "json"
            assert result.error is None
        finally:
            Path(temp_file).unlink()

    def test_parse_real_json_sample(self):
        """Test parsing a real JSON sample file."""
        parser = OpenAPIParser()
        result = parser.parse_file(
            "open-api-small-samples/3.0/json/petstore-simple.json"
        )

        assert result.success is True
        assert result.data is not None
        assert result.file_type == "json"
        assert result.error is None
        assert "openapi" in result.data
        assert "info" in result.data

    def test_parse_real_yaml_sample(self):
        """Test parsing a real YAML sample file."""
        parser = OpenAPIParser()
        result = parser.parse_file(
            "open-api-small-samples/3.0/yaml/petstore-simple.yaml"
        )

        assert result.success is True
        assert result.data is not None
        assert result.file_type == "yaml"
        assert result.error is None
        assert "openapi" in result.data
        assert "info" in result.data

    def test_parse_invalid_json_file(self):
        """Test parsing an invalid JSON file."""
        invalid_json = '{"openapi": "3.0.0", "info": {'  # Missing closing braces

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(invalid_json)
            temp_file = f.name

        try:
            parser = OpenAPIParser()
            result = parser.parse_file(temp_file)

            assert result.success is False
            assert result.data is None
            assert result.file_type == "json"
            assert "Invalid JSON format" in result.error
        finally:
            Path(temp_file).unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        parser = OpenAPIParser()
        result = parser.parse_file("/path/that/does/not/exist.json")

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert "File not found" in result.error

    def test_parse_unsupported_file_extension(self):
        """Test parsing a file with unsupported extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            temp_file = f.name

        try:
            parser = OpenAPIParser()
            result = parser.parse_file(temp_file)

            assert result.success is False
            assert result.data is None
            assert "Unsupported file extension" in result.error
        finally:
            Path(temp_file).unlink()

    def test_parse_with_path_object(self):
        """Test parsing using a Path object instead of string."""
        parser = OpenAPIParser()
        result = parser.parse_file(
            Path("open-api-small-samples/3.0/json/petstore-simple.json")
        )

        assert result.success is True
        assert result.data is not None
        assert result.file_type == "json"
