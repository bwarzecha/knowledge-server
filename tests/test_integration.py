"""Integration tests for the complete OpenAPI Processor."""

from src.cli.config import Config
from src.openapi_processor.processor import OpenAPIProcessor


class TestOpenAPIProcessorIntegration:
    """Integration tests for the complete pipeline."""

    def test_process_simple_sample_directory(self):
        """Test processing a simple sample with known structure."""
        config = Config()
        processor = OpenAPIProcessor(config)

        # Process just one simple file by targeting a specific sample
        chunks = processor.process_directory("open-api-small-samples/3.0/json/openapi-workshop")

        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            # Every chunk should have required fields
            assert "id" in chunk
            assert "document" in chunk
            assert "metadata" in chunk

            # Metadata should have required fields
            metadata = chunk["metadata"]
            assert "type" in metadata
            assert "source_file" in metadata
            assert "ref_ids" in metadata
            assert "referenced_by" in metadata
            assert "api_info_ref" in metadata
            assert "api_tags_ref" in metadata

            # Document should be valid YAML-formatted string
            assert isinstance(chunk["document"], str)
            assert len(chunk["document"]) > 0

            # ID should follow the pattern
            assert ":" in chunk["id"]

        # Check that we have different types of chunks
        chunk_types = {chunk["metadata"]["type"] for chunk in chunks}
        assert "info" in chunk_types  # Should always have info chunks

    def test_reference_resolution_works(self):
        """Test that references are properly resolved between chunks."""
        config = Config()
        processor = OpenAPIProcessor(config)

        # Use a sample with known references
        chunks = processor.process_directory("open-api-small-samples/3.0/json/openapi-workshop")

        # Find chunks with references
        chunks_with_refs = [c for c in chunks if c["metadata"]["ref_ids"]]

        if chunks_with_refs:
            # Verify references point to actual chunk IDs
            all_chunk_ids = {c["id"] for c in chunks}

            for chunk in chunks_with_refs:
                for ref_id in chunk["metadata"]["ref_ids"].keys():
                    # Reference should point to an actual chunk in the same batch
                    # (This validates the reference resolution is working)
                    if ref_id in all_chunk_ids:
                        # Find the referenced chunk
                        referenced_chunk = next(c for c in chunks if c["id"] == ref_id)

                        # The referenced chunk should have this chunk in its referenced_by list
                        assert chunk["id"] in referenced_chunk["metadata"]["referenced_by"]

    def test_content_preservation(self):
        """Test that original OpenAPI content is preserved in chunks."""
        config = Config()
        processor = OpenAPIProcessor(config)

        chunks = processor.process_directory("open-api-small-samples/3.0/json/openapi-workshop")

        # Find an info chunk and verify it contains expected info fields
        info_chunks = [c for c in chunks if c["metadata"]["type"] == "info"]
        assert len(info_chunks) > 0

        info_chunk = info_chunks[0]

        # Document should contain info content
        document = info_chunk["document"]
        assert "title:" in document or "version:" in document

        # Should be readable YAML format
        import yaml

        parsed_doc = yaml.safe_load(document)
        assert isinstance(parsed_doc, dict)

    def test_different_file_types(self):
        """Test processing both JSON and YAML files."""
        config = Config()
        processor = OpenAPIProcessor(config)

        # Test JSON files
        json_chunks = processor.process_directory("open-api-small-samples/3.0/json/openapi-workshop")

        # Test YAML files
        yaml_chunks = processor.process_directory("open-api-small-samples/3.0/yaml")

        # Both should produce chunks
        assert len(json_chunks) > 0
        assert len(yaml_chunks) > 0

        # Structure should be the same regardless of source format
        for chunks in [json_chunks, yaml_chunks]:
            for chunk in chunks:
                assert "id" in chunk
                assert "document" in chunk
                assert "metadata" in chunk
