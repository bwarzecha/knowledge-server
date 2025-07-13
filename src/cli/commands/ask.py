"""Ask command - test searchAPI functionality from command line."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.config import Config
from src.mcp_server.searchAPI import searchAPI
from src.mcp_server.shared_resources import initialize_shared_resources

logger = logging.getLogger(__name__)


def ask_command(
    config: Config,
    query: str,
    max_response_length: int = 4000,
    max_chunks: int = 50,
    include_references: bool = True,
    max_depth: int = 3,
    verbose: bool = False,
):
    """Ask questions about API documentation using the searchAPI functionality."""
    asyncio.run(
        _ask_command_async(
            config,
            query,
            max_response_length,
            max_chunks,
            include_references,
            max_depth,
            verbose,
        )
    )


async def _ask_command_async(
    config: Config,
    query: str,
    max_response_length: int,
    max_chunks: int,
    include_references: bool,
    max_depth: int,
    verbose: bool,
):
    """Async implementation of ask command."""

    if verbose:
        logger.info(f"🔍 Query: {query}")
        logger.info(f"⚙️  Settings: max_response_length={max_response_length}, max_chunks={max_chunks}")
        logger.info(f"🔗 References: {'enabled' if include_references else 'disabled'}, max_depth={max_depth}")
        logger.info("")

    # Check if indices exist
    if not Path(config.vector_store_dir).exists():
        logger.error(f"❌ Vector store not found at {config.vector_store_dir}")
        logger.error("💡 Run 'knowledge-server index' first to build the indices")
        sys.exit(1)

    if not Path(config.api_index_path).exists():
        logger.error(f"❌ API index not found at {config.api_index_path}")
        logger.error("💡 Run 'knowledge-server index' first to build the indices")
        sys.exit(1)

    if verbose:
        logger.info("✅ Pre-built indices found")
        logger.info("🔧 Initializing shared resources...")

    # Initialize shared resources (same as MCP server)
    try:
        initialize_shared_resources(config)
        if verbose:
            logger.info("✅ Shared resources loaded")
            logger.info("🚀 Processing query...")
            logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to initialize resources: {e}")
        sys.exit(1)

    # Call searchAPI with the provided parameters
    try:
        response = await searchAPI(
            query=query,
            max_response_length=max_response_length,
            max_chunks=max_chunks,
            include_references=include_references,
            max_depth=max_depth,
        )

        # Print the response to stdout (clean output)
        print(response)

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        sys.exit(1)
