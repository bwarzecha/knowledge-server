"""Test to show sample output from the markdown processor."""

import json
from pathlib import Path

from src.markdown_processor.chunk_assembler import ChunkAssembler
from src.markdown_processor.header_extractor import HeaderExtractor
from src.markdown_processor.parser import MarkdownParser
from src.markdown_processor.section_splitter import (AdaptiveSectionSplitter,
                                                     SplittingConfig)


def test_show_sample_output():
    """Show sample output from processing a markdown file."""
    # Setup components
    parser = MarkdownParser()
    header_extractor = HeaderExtractor()
    splitter = AdaptiveSectionSplitter(SplittingConfig(max_tokens=500))
    assembler = ChunkAssembler()

    # Process sample file
    samples_dir = Path(__file__).parent.parent.parent / "samples"
    sample_file = samples_dir / "Amazon Advertising Advanced Tools Center.md"

    # Parse file
    parse_result = parser.parse_file(sample_file)
    assert parse_result.success

    # Extract headers
    headers = header_extractor.extract_headers(parse_result.content)

    # Split sections
    sections = splitter.split_content(
        parse_result.content, headers, parse_result.frontmatter
    )

    # Assemble chunks
    chunks = assembler.assemble_chunks(
        sections, parse_result.frontmatter, str(sample_file.name)
    )

    print("\n\n=== SAMPLE OUTPUT ===")
    print(f"File: {sample_file.name}")
    print(f"Frontmatter: {bool(parse_result.frontmatter)}")
    print(f"Headers found: {len(headers)}")
    print(f"Sections created: {len(sections)}")
    print(f"Chunks generated: {len(chunks)}")

    print("\n--- First 3 Chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Type: {chunk['metadata']['type']}")
        print(f"  Title: {chunk['metadata']['title']}")
        print(f"  Word Count: {chunk['metadata']['word_count']}")
        print(f"  Token Count: {sections[i].token_count}")
        print(f"  Has Tables: {chunk['metadata']['has_tables']}")
        print(f"  External Links: {len(chunk['metadata']['external_links'])}")
        print(f"  Document Preview: {chunk['document'][:100]}...")

        if chunk["metadata"].get("is_split_section"):
            print(
                f"  Split Info: {chunk['metadata']['split_index']}/{chunk['metadata']['total_splits']}"
            )

    # Show navigation example
    print("\n--- Navigation Example ---")
    second_chunk = chunks[1] if len(chunks) > 1 else chunks[0]
    nav_meta = second_chunk["metadata"]
    print(f"Chunk: {second_chunk['id']}")
    print(f"  Previous: {nav_meta.get('previous_chunk', 'None')}")
    print(f"  Next: {nav_meta.get('next_chunk', 'None')}")
    print(f"  Parent: {nav_meta.get('parent_section', 'None')}")
    print(f"  Children: {nav_meta.get('child_sections', [])}")

    # Validate chunks
    errors = assembler.validate_chunks(chunks)
    print("\n--- Validation ---")
    print(f"Validation errors: {len(errors)}")
    if errors:
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    else:
        print("  ✓ All chunks valid!")

    # Show chunk structure as JSON
    print("\n--- Sample Chunk Structure (JSON) ---")
    sample_chunk = chunks[0].copy()
    # Truncate document for display
    sample_chunk["document"] = sample_chunk["document"][:200] + "..."
    print(json.dumps(sample_chunk, indent=2, default=str))


if __name__ == "__main__":
    test_show_sample_output()
