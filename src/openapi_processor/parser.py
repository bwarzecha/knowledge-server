"""OpenAPI Parser for JSON and YAML files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass
class ParseResult:
    """Result of parsing an OpenAPI file."""

    success: bool
    data: Dict[str, Any] = None
    error: str = None
    file_type: str = None  # 'json' or 'yaml'


class OpenAPIParser:
    """Parses OpenAPI specification files in JSON or YAML format."""

    def parse_file(self, file_path: Union[str, Path]) -> ParseResult:
        """
        Parse an OpenAPI specification file.

        Args:
            file_path: Path to the OpenAPI file (.json, .yaml, .yml)

        Returns:
            ParseResult with parsed data or error information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return ParseResult(success=False, error=f"File not found: {file_path}")

        if not file_path.is_file():
            return ParseResult(success=False, error=f"Path is not a file: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            return ParseResult(success=False, error=f"Unable to read file as UTF-8: {e}")
        except Exception as e:
            return ParseResult(success=False, error=f"Error reading file: {e}")

        # Determine file type by extension
        file_type = self._get_file_type(file_path)

        # Parse based on file type
        if file_type == "json":
            return self._parse_json(content, file_type)
        elif file_type == "yaml":
            return self._parse_yaml(content, file_type)
        else:
            return ParseResult(success=False, error=f"Unsupported file extension: {file_path.suffix}")

    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        extension = file_path.suffix.lower()
        if extension == ".json":
            return "json"
        elif extension in [".yaml", ".yml"]:
            return "yaml"
        else:
            return "unknown"

    def _parse_json(self, content: str, file_type: str) -> ParseResult:
        """Parse JSON content."""
        try:
            data = json.loads(content)
            return ParseResult(success=True, data=data, file_type=file_type)
        except json.JSONDecodeError as e:
            return ParseResult(success=False, error=f"Invalid JSON format: {e}", file_type=file_type)
        except Exception as e:
            return ParseResult(success=False, error=f"Error parsing JSON: {e}", file_type=file_type)

    def _parse_yaml(self, content: str, file_type: str) -> ParseResult:
        """Parse YAML content."""
        try:
            data = yaml.safe_load(content)
            return ParseResult(success=True, data=data, file_type=file_type)
        except yaml.YAMLError as e:
            return ParseResult(success=False, error=f"Invalid YAML format: {e}", file_type=file_type)
        except Exception as e:
            return ParseResult(success=False, error=f"Error parsing YAML: {e}", file_type=file_type)
