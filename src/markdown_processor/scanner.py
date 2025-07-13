"""Directory Scanner for Markdown files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


@dataclass
class ScannerConfig:
    """Configuration for the directory scanner."""

    skip_hidden_files: bool = True
    supported_extensions: List[str] = None
    process_readme_files: bool = True

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [".md", ".markdown"]


class DirectoryScanner:
    """Scans directories for Markdown files."""

    def __init__(self, config: ScannerConfig = None):
        self.config = config or ScannerConfig()

    def scan_for_markdown_files(self, root_dir: str) -> Iterator[str]:
        """
        Recursively scan for Markdown files.

        Args:
            root_dir: Root directory path to scan

        Yields:
            Relative file paths with extensions (.md, .markdown)
        """
        root_path = Path(root_dir)

        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        if not root_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {root_dir}")

        for file_path in self._walk_directory(root_path):
            relative_path = file_path.relative_to(root_path)
            yield str(relative_path)

    def _walk_directory(self, path: Path) -> Iterator[Path]:
        """Recursively walk directory tree and yield matching files."""
        try:
            for item in path.iterdir():
                # Skip hidden files/directories if configured
                if self.config.skip_hidden_files and item.name.startswith("."):
                    continue

                if item.is_file():
                    if self._is_markdown_file(item):
                        yield item
                elif item.is_dir():
                    yield from self._walk_directory(item)
        except PermissionError:
            # Log and skip directories we can't access
            pass

    def _is_markdown_file(self, file_path: Path) -> bool:
        """Check if file has a supported Markdown extension."""
        return file_path.suffix.lower() in self.config.supported_extensions
