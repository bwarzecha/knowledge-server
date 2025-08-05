#!/usr/bin/env python3
"""Build API index for query expansion - one entry per file."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import tiktoken

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.openapi_processor.parser import OpenAPIParser  # noqa: E402
from src.openapi_processor.scanner import DirectoryScanner  # noqa: E402


class IndexBuilder:
    def __init__(self, max_endpoints_per_file: int = 20):
        self.scanner = DirectoryScanner()
        self.parser = OpenAPIParser()
        self.max_endpoints_per_file = max_endpoints_per_file
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def build_index(self, directories: List[str]) -> List[Dict[str, Any]]:
        """Build structured API index - one entry per file."""
        file_entries = []

        for directory in directories:
            for relative_path in self.scanner.scan_for_openapi_files(directory):
                file_path = Path(directory) / relative_path
                file_entry = self._process_file(file_path, relative_path)
                if file_entry:
                    file_entries.append(file_entry)

        return file_entries

    def _process_file(self, file_path: Path, relative_path: str) -> Dict[str, Any]:
        """Extract all endpoints from a single file into one text entry."""
        result = self.parser.parse_file(file_path)
        if not result.success:
            return None

        spec = result.data
        info = spec.get("info", {})

        # Create full text representation directly
        text_lines = []

        # File header with metadata
        file_name = Path(relative_path).stem[:12]
        title = info.get("title", "")

        header = f"API: {file_name}"
        if title:
            header += f" - {title[:30]}"
        text_lines.append(header)

        # Add global tags if available
        global_tags = self._extract_global_tags(spec.get("tags", []))
        if global_tags:
            text_lines.append(f"Tags: {', '.join(global_tags[:5])}")

        # Add description if available and useful
        description = info.get("description", "").strip()
        if description and len(description) > 10:
            # Take first meaningful sentence
            desc_snippet = description.split(".")[0][:80]
            if desc_snippet:
                text_lines.append(f"Description: {desc_snippet}")

        # Extract endpoints as text
        paths = spec.get("paths", {})
        endpoint_count = 0

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue

                if endpoint_count >= self.max_endpoints_per_file:
                    break

                endpoint_text = self._create_endpoint_text(path, method, operation)
                if endpoint_text:
                    text_lines.append(endpoint_text)
                    endpoint_count += 1

        # Return as simple structure
        full_text = "\n".join(text_lines)

        return {"file": relative_path, "text": full_text}

    def _extract_global_tags(self, tags_list: list, max_tags: int = 5) -> list:
        """Extract global API tags."""
        tag_names = []
        for tag in tags_list:
            if isinstance(tag, dict):
                name = tag.get("name", "").strip()
                if name:
                    tag_names.append(name)
            elif isinstance(tag, str):
                tag_names.append(tag.strip())
        return tag_names[:max_tags]

    def _create_endpoint_text(self, path: str, method: str, operation: dict) -> str:
        """Create compact text representation of endpoint."""
        parts = [f"{method.upper()} {path}"]

        # Add operation ID if available
        operation_id = operation.get("operationId", "")
        if operation_id:
            parts.append(f"[{operation_id}]")

        # Add summary if available and useful
        summary = operation.get("summary", "").strip()
        if summary and len(summary) > 5:
            summary_short = summary[:40]
            if len(summary) > 40:
                summary_short += "..."
            parts.append(f"({summary_short})")

        # Add operation tags
        op_tags = operation.get("tags", [])
        if op_tags:
            tags_str = ",".join(op_tags[:2])  # Limit to first 2 tags
            parts.append(f"{{{tags_str}}}")

        return " ".join(parts)

    def save_index(
        self,
        file_entries: List[Dict[str, Any]],
        output_path: str = "data/api_index.json",
    ):
        """Save text-based index to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON with simple structure
        with open(output_file, "w") as f:
            json.dump(
                {
                    "version": "2.0",  # Version 2.0 for text-based format
                    "total_files": len(file_entries),
                    "files": file_entries,
                },
                f,
                indent=2,
            )

        # Calculate total text size
        total_text = "\n\n".join(entry["text"] for entry in file_entries)
        tokens = len(self.tokenizer.encode(total_text))

        logger.info(f"✅ Index saved to {output_path}")
        logger.info(f"   Files: {len(file_entries)}")
        logger.info(f"   Total text: {len(total_text):,} chars, {tokens:,} tokens")
        return tokens

    def create_compact_text(
        self, file_entries: List[Dict[str, Any]], max_chars: int = 1200
    ) -> str:
        """Create compact text representation for LLM context."""
        combined_texts = []
        char_count = 0

        for file_entry in file_entries:
            file_text = file_entry["text"]

            # Check if adding this file would exceed the limit
            test_length = char_count + len(file_text) + 2  # +2 for newlines

            if test_length > max_chars:
                # If we haven't added any content yet, take partial content from first file
                if not combined_texts:
                    # Take as much as we can from the first file
                    available_chars = max_chars - 10  # Leave some buffer
                    truncated_text = file_text[:available_chars]
                    # Try to cut at a line boundary
                    if "\n" in truncated_text:
                        lines = truncated_text.split("\n")
                        truncated_text = "\n".join(
                            lines[:-1]
                        )  # Remove partial last line
                    combined_texts.append(truncated_text)
                break

            combined_texts.append(file_text)
            char_count = test_length

        return "\n\n".join(combined_texts)


def main():
    """Build and save API index."""
    builder = IndexBuilder()

    # Build from samples directory
    directories = ["samples"]
    logger.info(f"🔄 Building text-based index from: {directories}")

    file_entries = builder.build_index(directories)
    builder.save_index(file_entries)

    # Show sample entries
    logger.info("\n📄 Sample file entries (first 2):")
    for entry in file_entries[:2]:
        lines = entry["text"].split("\n")
        logger.info(f"   {entry['file']}: {lines[0]} - {len(lines)} lines")

    # Test compact representation
    compact = builder.create_compact_text(file_entries)
    logger.info(f"\n📊 Compact representation ({len(compact)} chars):")
    logger.info(compact[:500] + "..." if len(compact) > 500 else compact)


if __name__ == "__main__":
    main()
