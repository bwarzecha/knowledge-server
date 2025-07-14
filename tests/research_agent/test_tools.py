"""Tests for Research Agent tools with real OpenAPI data."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.cli.config import Config
from src.openapi_processor.processor import OpenAPIProcessor
from src.research_agent.tools import generate_api_context, getChunks, searchChunks
from src.vector_store.vector_store_manager import VectorStoreManager


class TestResearchAgentTools:
    """Test Research Agent tools with real OpenAPI specifications."""

    @pytest.fixture(scope="class")
    def temp_environment(self):
        """Setup test environment with multiple OpenAPI specs indexed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            specs_dir = temp_path / "specs"
            specs_dir.mkdir()
            vector_dir = temp_path / "vector"

            # Copy test specs to temp directory
            test_specs = [
                "open-api-small-samples/3.0/json/petstore-simple.json",
                "open-api-small-samples/3.0/json/complex-nesting.json",
                "open-api-small-samples/3.0/json/circular.json",
                "open-api-small-samples/3.1/json/train-travel.json",
            ]

            for spec in test_specs:
                shutil.copy(spec, specs_dir / Path(spec).name)

            # Also copy workshop directory for multi-file tests
            workshop_src = Path("open-api-small-samples/3.0/json/openapi-workshop")
            workshop_dst = specs_dir / "workshop"
            shutil.copytree(workshop_src, workshop_dst)

            # Process OpenAPI specs
            config = Config()
            processor = OpenAPIProcessor(config)
            chunks = processor.process_directory(str(specs_dir))

            # Setup vector store using config from .env
            vector_store = VectorStoreManager(
                persist_directory=str(vector_dir),
                collection_name="test_research_agent",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast model for testing
                embedding_device="mps",
                reset_on_start=True,
            )
            vector_store.setup()
            vector_store.add_chunks(chunks)

            # Create API context
            api_context = """Available API Files:
- petstore-simple.json: Simple Pet Store API (Basic CRUD operations)
- complex-nesting.json: Complex Schema Nesting Examples
- circular.json: Circular Reference Test API
- train-travel.json: Train Travel Booking API (Comprehensive)
- workshop/: OpenAPI Workshop Examples (Multiple files)"""

            yield {
                "vector_store": vector_store,
                "api_context": api_context,
                "chunks_count": len(chunks),
                "specs_dir": str(specs_dir),
            }

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_search_chunks_basic_functionality(self, temp_environment):
        """Test basic searchChunks functionality with simple query."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Execute
        actual = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="pet operations",
            max_chunks=5,
        )

        # Expected
        expected_structure = {
            "has_results": True,
            "max_chunks": 5,
            "has_files_searched": True,
            "has_api_context": True,
        }

        # Assert
        assert actual.total_found > 0
        assert len(actual.chunks) <= expected_structure["max_chunks"]
        assert actual.search_time_ms > 0
        assert len(actual.files_searched) > 0
        assert "Pet Store API" in actual.api_context

        # Verify chunk structure
        for chunk in actual.chunks:
            assert chunk.chunk_id
            assert chunk.title
            assert chunk.content_preview
            assert chunk.chunk_type in [
                "operation",
                "component",
                "schema",
                "info",
                "unknown",
            ]
            assert chunk.file_name
            assert 0.0 <= chunk.relevance_score <= 1.0
            assert isinstance(chunk.ref_ids, list)

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_search_chunks_with_references_included(self, temp_environment):
        """Test searchChunks with reference IDs included."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Execute
        actual = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="complex schema",
            max_chunks=3,
            include_references=True,
        )

        # Expected
        expected = {"should_have_references": True, "max_results": 3}

        # Assert
        assert actual.total_found > 0
        assert len(actual.chunks) <= expected["max_results"]

        # At least some chunks should have references
        chunks_with_refs = [chunk for chunk in actual.chunks if len(chunk.ref_ids) > 0]
        assert len(chunks_with_refs) > 0

        # Check ref_ids structure
        for chunk in chunks_with_refs:
            for ref_id in chunk.ref_ids:
                assert isinstance(ref_id, str)
                assert ":" in ref_id  # Should be file:path format

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_search_chunks_file_filtering(self, temp_environment):
        """Test searchChunks with file filtering capability."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Execute - search with filter
        actual_filtered = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="API",
            max_chunks=5,
            file_filter="train-travel",
        )

        # Execute - search without filter
        actual_unfiltered = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="API",
            max_chunks=5,
            file_filter=None,
        )

        # Expected: filtered results should be smaller and match the filter

        # Assert
        assert actual_filtered.total_found >= 0
        assert actual_unfiltered.total_found >= actual_filtered.total_found

        # Filtered results should only come from train-travel files if any found
        if actual_filtered.total_found > 0:
            for file_name in actual_filtered.files_searched:
                assert "train-travel" in file_name or "train" in file_name

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_search_chunks_no_matching_filter(self, temp_environment):
        """Test searchChunks with non-matching file filter."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Execute
        actual = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="pet",
            max_chunks=5,
            file_filter="nonexistent-api",
        )

        # Expected
        expected = {"should_be_empty": 0}

        # Assert
        assert actual.total_found == expected["should_be_empty"]
        assert len(actual.chunks) == 0
        assert len(actual.files_searched) == 0

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_get_chunks_basic_retrieval(self, temp_environment):
        """Test basic getChunks functionality without expansion."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Get some chunk IDs first
        search_results = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="pet",
            max_chunks=3,
        )
        chunk_ids = [chunk.chunk_id for chunk in search_results.chunks[:2]]

        # Execute
        actual = await getChunks(vector_store=vector_store, chunk_ids=chunk_ids, expand_depth=0)

        # Expected
        expected = {
            "total_chunks": len(chunk_ids),
            "requested_count": len(chunk_ids),
            "expanded_count": 0,
            "expansion_depth_0": len(chunk_ids),
        }

        # Assert
        assert actual.total_chunks == expected["total_chunks"]
        assert len(actual.requested_chunks) == expected["requested_count"]
        assert len(actual.expanded_chunks) == expected["expanded_count"]
        assert actual.total_tokens > 0
        assert actual.expansion_stats[0] == expected["expansion_depth_0"]
        assert not actual.truncated

        # Check chunk structure
        for chunk in actual.requested_chunks:
            assert chunk.chunk_id in chunk_ids
            assert chunk.content
            assert chunk.source == "requested"
            assert chunk.expansion_depth == 0
            assert isinstance(chunk.ref_ids, list)
            assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_get_chunks_with_reference_expansion(self, temp_environment):
        """Test getChunks with reference expansion."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Find a chunk with references
        search_results = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="schema",
            max_chunks=5,
            include_references=True,
        )

        chunk_with_refs = None
        for chunk in search_results.chunks:
            if len(chunk.ref_ids) > 0:
                chunk_with_refs = chunk
                break

        if chunk_with_refs is None:
            pytest.skip("No chunks with references found for expansion test")

        # Execute
        actual = await getChunks(
            vector_store=vector_store,
            chunk_ids=[chunk_with_refs.chunk_id],
            expand_depth=2,
            max_total_chunks=10,
        )

        # Expected
        expected = {"should_expand": True, "requested_count": 1, "max_depth": 2}

        # Assert
        assert actual.total_chunks >= expected["requested_count"]
        assert len(actual.requested_chunks) == expected["requested_count"]
        assert actual.total_tokens > 0
        assert 0 in actual.expansion_stats
        assert actual.expansion_stats[0] == 1

        # Check expansion if it occurred
        if len(actual.expanded_chunks) > 0:
            for chunk in actual.expanded_chunks:
                assert chunk.source == "expanded"
                assert chunk.expansion_depth > 0
                assert chunk.expansion_depth <= expected["max_depth"]

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_get_chunks_truncation_behavior(self, temp_environment):
        """Test getChunks truncation with max_total_chunks limit."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Get a chunk with references for expansion
        search_results = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="complex",
            max_chunks=1,
            include_references=True,
        )

        chunk_with_refs = None
        for chunk in search_results.chunks:
            if len(chunk.ref_ids) > 0:
                chunk_with_refs = chunk
                break

        if chunk_with_refs is None:
            pytest.skip("No chunks with references found for truncation test")

        # Execute with very low limit
        actual = await getChunks(
            vector_store=vector_store,
            chunk_ids=[chunk_with_refs.chunk_id],
            expand_depth=3,
            max_total_chunks=2,  # Very restrictive
        )

        # Expected
        expected = {"max_total": 2, "should_respect_limit": True}

        # Assert
        assert actual.total_chunks <= expected["max_total"]
        assert len(actual.requested_chunks) + len(actual.expanded_chunks) <= expected["max_total"]

    @pytest.mark.asyncio
    async def test_get_chunks_empty_input(self, temp_environment):
        """Test getChunks with empty input."""
        # Setup
        vector_store = temp_environment["vector_store"]

        # Execute
        actual = await getChunks(vector_store=vector_store, chunk_ids=[], expand_depth=0)

        # Expected
        expected = {
            "total_chunks": 0,
            "requested_chunks": 0,
            "expanded_chunks": 0,
            "total_tokens": 0,
            "truncated": False,
        }

        # Assert
        assert actual.total_chunks == expected["total_chunks"]
        assert len(actual.requested_chunks) == expected["requested_chunks"]
        assert len(actual.expanded_chunks) == expected["expanded_chunks"]
        assert actual.total_tokens == expected["total_tokens"]
        assert actual.truncated == expected["truncated"]

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_get_chunks_circular_reference_protection(self, temp_environment):
        """Test getChunks handles circular references properly."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Search for chunks from the circular.json file
        search_results = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="circular",
            max_chunks=1,
            file_filter="circular",
        )

        if search_results.total_found == 0:
            pytest.skip("No circular reference chunks found")

        chunk_id = search_results.chunks[0].chunk_id

        # Execute with deep expansion that could trigger circular refs
        actual = await getChunks(
            vector_store=vector_store,
            chunk_ids=[chunk_id],
            expand_depth=5,  # Deep enough to trigger issues if not protected
            max_total_chunks=50,
        )

        # Expected: should complete without infinite loop

        # Assert - mainly that it completes and doesn't infinite loop
        assert actual.total_chunks > 0
        assert len(actual.requested_chunks) > 0
        assert actual.total_tokens > 0
        # The test completing is the main success criteria

    def test_generate_api_context_with_real_file(self, temp_environment):
        """Test generate_api_context with a real API index file."""
        # Setup
        specs_dir = temp_environment["specs_dir"]
        api_index_path = Path(specs_dir) / "api_index.json"

        api_index = {
            "petstore-simple.json": {"description": "Simple Pet Store API (Basic CRUD operations)"},
            "train-travel.json": {"description": "Train Travel Booking API (Comprehensive)"},
        }

        with open(api_index_path, "w") as f:
            json.dump(api_index, f)

        # Execute
        actual = generate_api_context(str(api_index_path))

        # Expected
        expected = {
            "should_contain_files": ["petstore-simple.json", "train-travel.json"],
            "should_contain_descriptions": ["Simple Pet Store", "Train Travel"],
        }

        # Assert
        assert "Available API Files:" in actual
        for file_name in expected["should_contain_files"]:
            assert file_name in actual
        for description in expected["should_contain_descriptions"]:
            assert description in actual

    def test_generate_api_context_missing_file(self):
        """Test generate_api_context with missing file."""
        # Setup
        nonexistent_path = "/nonexistent/path/api_index.json"

        # Execute
        actual = generate_api_context(nonexistent_path)

        # Expected
        expected = "Available API Files: (context unavailable)"

        # Assert
        assert actual == expected

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_complex_nesting_deep_expansion(self, temp_environment):
        """Test deep schema expansion with complex-nesting.json."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Search for complex nesting schemas
        search_results = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="array object",
            max_chunks=2,
            file_filter="complex-nesting",
            include_references=True,
        )

        if search_results.total_found == 0:
            pytest.skip("No complex nesting chunks found")

        chunk_with_refs = None
        for chunk in search_results.chunks:
            if len(chunk.ref_ids) > 0:
                chunk_with_refs = chunk
                break

        if chunk_with_refs is None:
            pytest.skip("No chunks with references in complex-nesting")

        # Execute with deep expansion
        actual = await getChunks(
            vector_store=vector_store,
            chunk_ids=[chunk_with_refs.chunk_id],
            expand_depth=5,  # Deep expansion for complex nesting
            max_total_chunks=20,
        )

        # Expected: should handle deep nesting well

        # Assert
        assert actual.total_chunks > 1  # Should expand
        assert len(actual.requested_chunks) == 1
        assert actual.total_tokens > 0

        # Check that we have expansion at multiple depths if available
        depth_levels = set(chunk.expansion_depth for chunk in actual.expanded_chunks)
        if len(actual.expanded_chunks) > 1:
            assert len(depth_levels) >= 1  # At least some depth variation

    @pytest.mark.asyncio
    @pytest.mark.bedrock
    async def test_multi_file_workshop_search(self, temp_environment):
        """Test search across multiple workshop files."""
        # Setup
        vector_store = temp_environment["vector_store"]
        api_context = temp_environment["api_context"]

        # Execute search across workshop files
        actual = await searchChunks(
            vector_store=vector_store,
            api_context=api_context,
            query="get post",
            max_chunks=5,
            file_filter="workshop",
        )

        # Expected: should find multiple files and workshop files

        # Assert
        assert actual.total_found > 0

        # Should find results from workshop files
        workshop_files = [
            f
            for f in actual.files_searched
            if "workshop" in f or any(workshop_name in f for workshop_name in ["get", "post", "hoot"])
        ]
        assert len(workshop_files) > 0
