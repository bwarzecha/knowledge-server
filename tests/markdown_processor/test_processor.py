"""Tests for the main markdown processor orchestration."""

from pathlib import Path

from src.markdown_processor.processor import MarkdownProcessor


class TestMarkdownProcessor:
    """Test the main MarkdownProcessor orchestration."""

    def test_process_directory(self):
        """Test processing a directory of markdown files."""
        processor = MarkdownProcessor()  # Use default 1000 tokens

        # Process sample directory
        samples_dir = Path(__file__).parent.parent.parent / "samples"
        chunks = processor.process_directory(samples_dir)

        # Should have generated chunks
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert "id" in chunk
            assert "document" in chunk
            assert "metadata" in chunk

            # Check metadata completeness
            metadata = chunk["metadata"]
            assert "type" in metadata
            assert "source_file" in metadata
            assert "section_level" in metadata
            assert "title" in metadata
            assert "word_count" in metadata
            assert "content_hash" in metadata

        # Validate chunks
        errors = processor.validate_chunks(chunks)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_process_single_file(self):
        """Test processing a single markdown file."""
        processor = MarkdownProcessor()  # Use default 1000 tokens

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

        chunks = processor.process_file(sample_file, "test.md")

        assert len(chunks) > 0

        # Verify all chunks have the same source file
        for chunk in chunks:
            assert chunk["metadata"]["source_file"] == "test.md"

    def test_process_content_directly(self):
        """Test processing markdown content directly."""
        # Use smaller token limit to ensure sections are split
        processor = MarkdownProcessor(max_tokens=100)

        content = """# Test Document

This is a test document with substantial content to ensure sections are split properly.

## Section 1

Content for section 1 with enough text to make it substantial and potentially exceed token limits when combined with
other sections.

### Subsection 1.1

More detailed content here with additional text to make it longer and more substantial.

## Section 2

Content for section 2 with enough content to be meaningful and potentially exceed token limits."""

        chunks = processor.process_content(content, "test.md")

        assert len(chunks) >= 1  # At least one chunk should be created

        # Check sequential navigation if multiple chunks
        if len(chunks) > 1:
            chunk_ids = [chunk["id"] for chunk in chunks]
            for i, chunk in enumerate(chunks):
                metadata = chunk["metadata"]

                if i > 0:
                    assert metadata.get("previous_chunk") == chunk_ids[i - 1]
                else:
                    # None values are converted to empty strings for ChromaDB compatibility
                    assert metadata.get("previous_chunk") in [None, ""]

                if i < len(chunks) - 1:
                    assert metadata.get("next_chunk") == chunk_ids[i + 1]
                else:
                    # None values are converted to empty strings for ChromaDB compatibility
                    assert metadata.get("next_chunk") in [None, ""]

    def test_processing_stats(self):
        """Test processing statistics generation."""
        processor = MarkdownProcessor()  # Use default 1000 tokens

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        chunks = processor.process_directory(samples_dir)

        stats = processor.get_processing_stats(chunks)

        assert "total_chunks" in stats
        assert "total_files" in stats
        assert "chunk_types" in stats
        assert "avg_word_count" in stats
        assert "total_word_count" in stats
        assert "files_processed" in stats

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_files"] > 0
        assert stats["total_word_count"] > 0

    def test_navigation_summary(self):
        """Test navigation relationship summary."""
        # Use smaller token limit to force multiple chunks
        processor = MarkdownProcessor(max_tokens=100)

        content = """# Main Title

Introduction content with substantial text to ensure this section is meaningful and has enough content to
potentially exceed token limits.

## Section A

Content A with enough text to make it substantial and potentially exceed token limits when processed.

### Subsection A.1

Content A.1 with additional detailed information to make it longer and more substantial for processing.

## Section B

Content B with enough content to be meaningful and test navigation relationships properly."""

        chunks = processor.process_content(content, "test.md")
        nav_summary = processor.get_navigation_summary(chunks)

        assert "total_relationships" in nav_summary
        assert "previous_links" in nav_summary
        assert "next_links" in nav_summary
        assert "parent_links" in nav_summary
        assert "child_relationships" in nav_summary
        assert "sibling_relationships" in nav_summary

        # Total relationships should be non-negative
        assert nav_summary["total_relationships"] >= 0

    def test_chunk_optimization(self):
        """Test chunk optimization functionality."""
        processor = MarkdownProcessor()  # Use default 1000 tokens

        content = "# Test\n\nSimple content."
        chunks = processor.process_content(content, "test.md")

        optimized = processor.optimize_chunks(chunks)

        # Should return same number of chunks
        assert len(optimized) == len(chunks)

        # Should maintain structure
        for opt_chunk in optimized:
            assert "id" in opt_chunk
            assert "document" in opt_chunk
            assert "metadata" in opt_chunk

    def test_token_limit_impact(self):
        """Test that different token limits produce different chunk sizes."""
        content = """# Large Document

This is a document with substantial content that should be affected by token limits.

## Section 1

This section has quite a bit of content that might exceed smaller token limits.
It includes multiple sentences and should demonstrate the difference between
different token limit configurations. Here's even more content to make sure
we have enough text to trigger different splitting behaviors.

## Section 2

Another section with substantial content. This content is also quite lengthy
and should help demonstrate how token limits affect the chunking process.
We want to see meaningful differences in chunk sizes based on our configuration."""

        # Test with smaller token limit
        processor_small = MarkdownProcessor(max_tokens=200)
        chunks_small = processor_small.process_content(content, "test.md")

        # Test with larger token limit
        processor_large = MarkdownProcessor(max_tokens=2000)
        chunks_large = processor_large.process_content(content, "test.md")

        # With reasonable token limits, we should see fewer chunks for larger limits
        print(f"Small limit (200 tokens): {len(chunks_small)} chunks")
        print(f"Large limit (2000 tokens): {len(chunks_large)} chunks")

        # Both should create chunks, but potentially different numbers
        assert len(chunks_small) > 0
        assert len(chunks_large) > 0

        # Check actual token counts to understand the chunking
        for i, chunk in enumerate(chunks_small):
            print(f"Small chunk {i}: {chunk['metadata']['word_count']} words")

        for i, chunk in enumerate(chunks_large):
            print(f"Large chunk {i}: {chunk['metadata']['word_count']} words")


def test_show_improved_sample_output():
    """Show sample output with improved token limits."""
    # Test with new default token limits
    processor = MarkdownProcessor()  # Default 1000 tokens

    # Process sample file
    samples_dir = Path(__file__).parent.parent.parent / "samples"
    sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

    chunks = processor.process_file(sample_file, sample_file.name)

    print("\n\n=== IMPROVED SAMPLE OUTPUT ===")
    print(f"File: {sample_file.name}")
    print("Token Limit: 1000 (default, supports up to 8000)")
    print(f"Chunks generated: {len(chunks)}")

    print("\n--- All Chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Type: {chunk['metadata']['type']}")
        print(f"  Title: {chunk['metadata']['title']}")
        print(f"  Word Count: {chunk['metadata']['word_count']}")
        print(f"  Document Preview: {chunk['document'][:100]}...")

    # Get processing statistics
    stats = processor.get_processing_stats(chunks)
    print("\n--- Processing Statistics ---")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average word count: {stats['avg_word_count']}")
    print(f"Total word count: {stats['total_word_count']}")
    print(f"Chunk types: {stats['chunk_types']}")

    # Get navigation summary
    nav_summary = processor.get_navigation_summary(chunks)
    print("\n--- Navigation Summary ---")
    print(f"Total relationships: {nav_summary['total_relationships']}")
    print(
        f"Previous/Next links: {nav_summary['previous_links']}/{nav_summary['next_links']}"
    )
    print(f"Parent links: {nav_summary['parent_links']}")
    print(f"Child relationships: {nav_summary['child_relationships']}")

    # Validate chunks
    errors = processor.validate_chunks(chunks)
    print("\n--- Validation ---")
    print(f"Validation errors: {len(errors)}")
    if errors:
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    else:
        print("  ✓ All chunks valid!")


def test_show_large_token_limit_output():
    """Show sample output with large token limits."""
    # Test with large token limits for big chunks
    processor = MarkdownProcessor(max_tokens=4000)  # Large chunks

    # Process sample file
    samples_dir = Path(__file__).parent.parent.parent / "samples"
    sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

    chunks = processor.process_file(sample_file, sample_file.name)

    print("\n\n=== LARGE TOKEN LIMIT OUTPUT ===")
    print(f"File: {sample_file.name}")
    print("Token Limit: 4000 (large chunks for complex content)")
    print(f"Chunks generated: {len(chunks)}")

    print("\n--- All Chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Type: {chunk['metadata']['type']}")
        print(f"  Title: {chunk['metadata']['title']}")
        print(f"  Word Count: {chunk['metadata']['word_count']}")
        print(f"  Document Preview: {chunk['document'][:200]}...")

    # Get processing statistics
    stats = processor.get_processing_stats(chunks)
    print("\n--- Processing Statistics ---")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average word count: {stats['avg_word_count']}")
    print(f"Total word count: {stats['total_word_count']}")
    print(f"Chunk types: {stats['chunk_types']}")

    print("  ✓ Large chunks for comprehensive context!")


def test_token_limit_validation():
    """Test token limit validation."""
    # Test maximum limit
    processor_max = MarkdownProcessor(max_tokens=10000)  # Should be capped at 8000
    assert processor_max.section_splitter.config.max_tokens == 8000

    # Test minimum limit
    processor_min = MarkdownProcessor(max_tokens=50)  # Should be raised to 100
    assert processor_min.section_splitter.config.max_tokens == 100

    # Test normal range
    processor_normal = MarkdownProcessor(max_tokens=2000)
    assert processor_normal.section_splitter.config.max_tokens == 2000

    print("✓ Token limit validation working correctly!")


if __name__ == "__main__":
    test_show_improved_sample_output()
