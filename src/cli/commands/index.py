"""Index command - builds vector store and API index."""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.config import Config
from src.openapi_processor.processor import OpenAPIProcessor
from src.query_expansion.index_builder import IndexBuilder
from src.vector_store.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


def index_command(config: Config):
    """Build vector store and API index from OpenAPI specifications."""
    logger.info("🔄 Starting indexing process...")
    logger.info(f"📁 Source directory: {config.openapi_specs_dir}")
    logger.info(f"💾 Vector store: {config.vector_store_dir}")
    logger.info(f"📊 API index: {config.api_index_path}")

    # Ensure data directories exist
    config.ensure_data_dirs()

    try:
        # Step 1: Process OpenAPI specifications
        logger.info("📋 Step 1: Processing OpenAPI specifications...")
        processor = OpenAPIProcessor()
        chunks = processor.process_directory(config.openapi_specs_dir)

        if not chunks:
            logger.error("❌ No chunks generated from OpenAPI specifications")
            sys.exit(1)

        logger.info(f"✅ Generated {len(chunks)} chunks from OpenAPI specifications")

        # Step 2: Build vector store
        logger.info("🧠 Step 2: Building vector store...")
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
        logger.info(f"📊 Adding {len(chunks)} chunks to vector store...")
        vector_store.add_chunks(chunks, batch_size=25)
        logger.info("✅ Vector store built successfully")

        # Step 3: Build API index
        logger.info("🔗 Step 3: Building API index...")
        index_builder = IndexBuilder()
        file_entries = index_builder.build_index([config.openapi_specs_dir])
        tokens = index_builder.save_index(file_entries, config.api_index_path)
        logger.info("✅ API index built successfully")

        # Summary
        logger.info("🎉 Indexing completed successfully!")
        logger.info(f"📊 Vector store: {len(chunks)} chunks in {config.vector_store_dir}")
        logger.info(f"🔗 API index: {len(file_entries)} files, {tokens:,} tokens in {config.api_index_path}")
        logger.info("🚀 Ready to serve! Run: knowledge-server serve")

    except Exception as e:
        logger.error(f"❌ Indexing failed: {str(e)}")
        sys.exit(1)
