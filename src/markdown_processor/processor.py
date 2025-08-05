"""Main orchestration for the markdown processor component."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from .chunk_assembler import ChunkAssembler
from .content_analyzer import ContentAnalyzer
from .header_extractor import HeaderExtractor
from .navigation_builder import NavigationBuilder
from .parser import MarkdownParser
from .reference_scanner import ReferenceScanner
from .scanner import DirectoryScanner
from .section_splitter import AdaptiveSectionSplitter, SplittingConfig

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Main orchestrator for the 5-phase markdown processing pipeline."""

    def __init__(self, max_tokens: int = 1000, include_context_in_splits: bool = True):
        """
        Initialize the markdown processor.

        Args:
            max_tokens: Maximum tokens per chunk (default 1000, supports up to 8000)
            include_context_in_splits: Whether to include context in split sections
        """
        # Validate token limit
        if max_tokens > 8000:
            logger.warning(
                f"Token limit {max_tokens} exceeds recommended maximum of 8000, capping at 8000"
            )
            max_tokens = 8000
        elif max_tokens < 100:
            logger.warning(
                f"Token limit {max_tokens} too small, setting minimum of 100"
            )
            max_tokens = 100
        # Initialize all components
        self.scanner = DirectoryScanner()
        self.parser = MarkdownParser()
        self.header_extractor = HeaderExtractor()
        self.section_splitter = AdaptiveSectionSplitter(
            SplittingConfig(
                max_tokens=max_tokens,
                include_context_in_splits=include_context_in_splits,
            )
        )
        self.reference_scanner = ReferenceScanner()
        self.navigation_builder = NavigationBuilder()
        self.content_analyzer = ContentAnalyzer()
        self.chunk_assembler = ChunkAssembler()

    def process_directory(self, docs_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Main entry point - orchestrates the complete 5-phase pipeline.

        Args:
            docs_dir: Directory containing markdown files to process

        Returns:
            List of complete chunk dictionaries ready for vector storage
        """
        docs_dir = Path(docs_dir)
        if not docs_dir.exists():
            raise ValueError(f"Directory does not exist: {docs_dir}")

        logger.info(f"Starting markdown processing for directory: {docs_dir}")

        all_chunks = []
        processed_files = 0
        failed_files = 0

        # Phase 1: Directory scanning
        try:
            markdown_files = list(self.scanner.scan_for_markdown_files(str(docs_dir)))
            logger.info(f"Found {len(markdown_files)} markdown files to process")
        except Exception as e:
            logger.error(f"Failed to scan directory {docs_dir}: {e}")
            return []

        # Process each file through phases 2-5
        for file_path in markdown_files:
            try:
                full_path = docs_dir / file_path
                file_chunks = self.process_file(full_path, file_path)
                all_chunks.extend(file_chunks)
                processed_files += 1

                if processed_files % 10 == 0:
                    logger.info(f"Processed {processed_files} files...")

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                failed_files += 1
                continue

        logger.info(
            f"Processing complete: {processed_files} files processed, "
            f"{failed_files} files failed, {len(all_chunks)} chunks generated"
        )

        return all_chunks

    def process_file(
        self, file_path: Union[str, Path], relative_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single markdown file through phases 2-5.

        Args:
            file_path: Full path to the markdown file
            relative_path: Relative path for chunk IDs (defaults to filename)

        Returns:
            List of chunks generated from this file
        """
        file_path = Path(file_path)
        if relative_path is None:
            relative_path = file_path.name

        logger.debug(f"Processing file: {file_path}")

        # Phase 2: File parsing
        parse_result = self.parser.parse_file(file_path)
        if not parse_result.success:
            logger.warning(f"Failed to parse file {file_path}: {parse_result.error}")
            return []

        # Phase 3: Content extraction and section identification
        headers = self.header_extractor.extract_headers(parse_result.content)
        logger.debug(f"Extracted {len(headers)} headers from {file_path}")

        sections = self.section_splitter.split_content(
            parse_result.content, headers, parse_result.frontmatter
        )
        logger.debug(f"Created {len(sections)} sections from {file_path}")

        # Phase 4: Reference resolution and navigation building
        # (References are scanned per-section in chunk assembly)

        # Phase 5: Final assembly and output
        chunks = self.chunk_assembler.assemble_chunks(
            sections, parse_result.frontmatter, relative_path
        )

        logger.debug(f"Generated {len(chunks)} chunks from {file_path}")
        return chunks

    def process_files(
        self, file_paths: List[Union[str, Path]], base_dir: Union[str, Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple specific files.

        Args:
            file_paths: List of file paths to process
            base_dir: Base directory for calculating relative paths

        Returns:
            List of all chunks from all files
        """
        if base_dir:
            base_dir = Path(base_dir)

        all_chunks = []

        for file_path in file_paths:
            file_path = Path(file_path)

            # Calculate relative path
            if base_dir and file_path.is_absolute():
                try:
                    relative_path = str(file_path.relative_to(base_dir))
                except ValueError:
                    relative_path = file_path.name
            else:
                relative_path = file_path.name

            try:
                file_chunks = self.process_file(file_path, relative_path)
                all_chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue

        return all_chunks

    def process_content(
        self,
        content: str,
        filename: str = "content.md",
        frontmatter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process markdown content directly (without file I/O).

        Args:
            content: Markdown content to process
            filename: Filename to use for chunk IDs
            frontmatter: Optional frontmatter dictionary

        Returns:
            List of chunks generated from content
        """
        logger.debug(f"Processing content for {filename}")

        # Phase 3: Content extraction and section identification
        headers = self.header_extractor.extract_headers(content)
        sections = self.section_splitter.split_content(content, headers, frontmatter)

        # Phase 5: Final assembly and output
        chunks = self.chunk_assembler.assemble_chunks(sections, frontmatter, filename)

        logger.debug(f"Generated {len(chunks)} chunks from content")
        return chunks

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Validate generated chunks for completeness and consistency.

        Args:
            chunks: List of chunk dictionaries to validate

        Returns:
            List of validation errors (empty if valid)
        """
        return self.chunk_assembler.validate_chunks(chunks)

    def optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunks for better search performance.

        Args:
            chunks: List of chunk dictionaries to optimize

        Returns:
            Optimized chunks
        """
        return self.chunk_assembler.optimize_chunks(chunks)

    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks.

        Args:
            chunks: List of processed chunks

        Returns:
            Dictionary with processing statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "chunk_types": {},
                "avg_word_count": 0,
                "total_word_count": 0,
            }

        # Group by source file
        files = set()
        chunk_types = {}
        total_words = 0

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Track files
            source_file = metadata.get("source_file", "unknown")
            files.add(source_file)

            # Track chunk types
            chunk_type = metadata.get("type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Track word counts
            word_count = metadata.get("word_count", 0)
            total_words += word_count

        avg_words = total_words / len(chunks) if chunks else 0

        return {
            "total_chunks": len(chunks),
            "total_files": len(files),
            "chunk_types": chunk_types,
            "avg_word_count": round(avg_words, 1),
            "total_word_count": total_words,
            "files_processed": sorted(list(files)),
        }

    def get_navigation_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of navigation relationships in chunks.

        Args:
            chunks: List of processed chunks

        Returns:
            Dictionary with navigation statistics
        """
        if not chunks:
            return {"total_relationships": 0}

        total_prev = 0
        total_next = 0
        total_parent = 0
        total_children = 0
        total_siblings = 0

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            if metadata.get("previous_chunk"):
                total_prev += 1
            if metadata.get("next_chunk"):
                total_next += 1
            if metadata.get("parent_section"):
                total_parent += 1

            children = metadata.get("child_sections", [])
            total_children += len(children)

            siblings = metadata.get("sibling_sections", [])
            total_siblings += len(siblings)

        return {
            "total_relationships": total_prev
            + total_next
            + total_parent
            + total_children
            + total_siblings,
            "previous_links": total_prev,
            "next_links": total_next,
            "parent_links": total_parent,
            "child_relationships": total_children,
            "sibling_relationships": total_siblings,
        }
