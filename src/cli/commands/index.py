"""Index command - builds vector store and API index."""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.config import Config
from src.markdown_processor.processor import MarkdownProcessor
from src.openapi_processor.processor import OpenAPIProcessor
from src.query_expansion.index_builder import IndexBuilder
from src.vector_store.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


def index_command(
    config: Config,
    skip_openapi: bool = False,
    skip_markdown: bool = False,
    max_tokens: int = None,
    markdown_dir: str = None,
):
    """Build vector store and API index from OpenAPI specs and markdown files."""
    logger.info("🔄 Starting indexing process...")

    # Determine source directories
    openapi_dir = config.openapi_specs_dir
    md_dir = markdown_dir or config.markdown_source_dir

    logger.info(f"📁 OpenAPI directory: {openapi_dir} {'(skipped)' if skip_openapi else ''}")
    logger.info(f"📁 Markdown directory: {md_dir} {'(skipped)' if skip_markdown else ''}")
    logger.info(f"💾 Vector store: {config.vector_store_dir}")
    logger.info(f"📊 API index: {config.api_index_path}")

    # Validation
    if skip_openapi and skip_markdown:
        logger.error("❌ Cannot skip both OpenAPI and markdown processing")
        sys.exit(1)

    # Ensure data directories exist
    config.ensure_data_dirs()

    try:
        all_chunks = []
        all_source_dirs = []

        # Step 1: Process OpenAPI specifications (unless skipped)
        if not skip_openapi:
            logger.info("📋 Step 1: Processing OpenAPI specifications...")
            openapi_processor = OpenAPIProcessor()
            openapi_chunks = openapi_processor.process_directory(openapi_dir)

            if openapi_chunks:
                all_chunks.extend(openapi_chunks)
                all_source_dirs.append(openapi_dir)
                logger.info(f"✅ Generated {len(openapi_chunks)} chunks from OpenAPI specifications")
            else:
                logger.warning("⚠️ No chunks generated from OpenAPI specifications")
        else:
            logger.info("⏭️ Step 1: Skipping OpenAPI processing")

        # Step 2: Process Markdown files (unless skipped)
        if not skip_markdown:
            logger.info("📝 Step 2: Processing markdown files...")
            markdown_processor = MarkdownProcessor(max_tokens=max_tokens or config.markdown_max_tokens)
            markdown_chunks = markdown_processor.process_directory(md_dir)

            if markdown_chunks:
                all_chunks.extend(markdown_chunks)
                all_source_dirs.append(md_dir)
                logger.info(f"✅ Generated {len(markdown_chunks)} chunks from markdown files")
            else:
                logger.warning("⚠️ No chunks generated from markdown files")
        else:
            logger.info("⏭️ Step 2: Skipping markdown processing")

        # Check if we have any chunks
        if not all_chunks:
            logger.error("❌ No chunks generated from any source")
            sys.exit(1)

        logger.info(f"📊 Total chunks: {len(all_chunks)}")

        # Step 3: Build vector store
        logger.info("🧠 Step 3: Building vector store...")
        vector_store = VectorStoreManager(
            persist_directory=config.vector_store_dir,
            collection_name=config.vector_store_collection,
            embedding_model_name=config.embedding_model,
            embedding_device=config.embedding_device,
            max_tokens=config.max_tokens,
            reset_on_start=True,  # Clean slate for indexing
        )

        # Setup and add chunks
        vector_store.setup()
        logger.info(f"📊 Adding {len(all_chunks)} chunks to vector store...")
        vector_store.add_chunks(all_chunks, batch_size=25)
        logger.info("✅ Vector store built successfully")

        # Step 4: Build API index
        logger.info("🔗 Step 4: Building API index...")
        index_builder = IndexBuilder()
        file_entries = index_builder.build_index(all_source_dirs)
        tokens = index_builder.save_index(file_entries, config.api_index_path)
        logger.info("✅ API index built successfully")

        # Summary
        logger.info("🎉 Indexing completed successfully!")
        logger.info(f"📊 Vector store: {len(all_chunks)} chunks in {config.vector_store_dir}")
        logger.info(f"🔗 API index: {len(file_entries)} files, {tokens:,} tokens in {config.api_index_path}")
        logger.info("🚀 Ready to serve! Run: knowledge-server serve")

    except Exception as e:
        logger.error(f"❌ Indexing failed: {str(e)}")
        sys.exit(1)
