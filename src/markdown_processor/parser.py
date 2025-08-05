"""Markdown Parser for files with frontmatter and content."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import frontmatter


@dataclass
class ParseResult:
    """Result of parsing a Markdown file."""

    success: bool
    frontmatter: Dict[str, Any] = None
    content: str = None
    error: str = None
    has_frontmatter: bool = False


class MarkdownParser:
    """Parses Markdown files with optional YAML frontmatter."""

    def parse_file(self, file_path: Union[str, Path]) -> ParseResult:
        """
        Parse a Markdown file with optional frontmatter.

        Args:
            file_path: Path to the Markdown file (.md, .markdown)

        Returns:
            ParseResult with frontmatter, content, or error information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return ParseResult(success=False, error=f"File not found: {file_path}")

        if not file_path.is_file():
            return ParseResult(success=False, error=f"Path is not a file: {file_path}")

        try:
            # Use python-frontmatter to parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # Extract frontmatter and content
            frontmatter_data = post.metadata if post.metadata else {}
            content = post.content if post.content else ""
            has_frontmatter = bool(post.metadata)

            return ParseResult(
                success=True,
                frontmatter=frontmatter_data,
                content=content,
                has_frontmatter=has_frontmatter,
            )

        except UnicodeDecodeError as e:
            return ParseResult(
                success=False, error=f"Unable to read file as UTF-8: {e}"
            )
        except Exception as e:
            # Check if it's a YAML error by looking at the error message
            if "yaml" in str(e).lower() or "parser" in str(type(e).__name__).lower():
                return ParseResult(
                    success=False, error=f"Invalid YAML frontmatter: {e}"
                )
            return ParseResult(success=False, error=f"Error parsing file: {e}")

    def parse_content(self, content_text: str) -> ParseResult:
        """
        Parse markdown content string with optional frontmatter.

        Args:
            content_text: Markdown content as string

        Returns:
            ParseResult with frontmatter, content, or error information
        """
        try:
            post = frontmatter.loads(content_text)

            frontmatter_data = post.metadata if post.metadata else {}
            content = post.content if post.content else ""
            has_frontmatter = bool(post.metadata)

            return ParseResult(
                success=True,
                frontmatter=frontmatter_data,
                content=content,
                has_frontmatter=has_frontmatter,
            )

        except Exception as e:
            # Check if it's a YAML error by looking at the error message
            if "yaml" in str(e).lower() or "parser" in str(type(e).__name__).lower():
                return ParseResult(
                    success=False, error=f"Invalid YAML frontmatter: {e}"
                )
            return ParseResult(success=False, error=f"Error parsing content: {e}")
