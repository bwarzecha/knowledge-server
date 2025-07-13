"""Serve command - starts MCP server with pre-built indices."""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.config import Config

logger = logging.getLogger(__name__)


def serve_command(config: Config, verbose: bool = False):
    """Start MCP server using pre-built indices."""
    if verbose:
        logger.info(f"🚀 Starting {config.mcp_server_name}...")
        logger.info(f"💾 Vector store: {config.vector_store_dir}")
        logger.info(f"📊 API index: {config.api_index_path}")

    # Check if indices exist
    if not Path(config.vector_store_dir).exists():
        logger.error(f"Vector store not found at {config.vector_store_dir}")
        logger.error("Run 'knowledge-server index' first to build the indices")
        sys.exit(1)

    if not Path(config.api_index_path).exists():
        logger.error(f"API index not found at {config.api_index_path}")
        logger.error("Run 'knowledge-server index' first to build the indices")
        sys.exit(1)

    if verbose:
        logger.info("✅ Pre-built indices found")

    # Import and start MCP server
    try:
        from src.mcp_server.server import start_server

        start_server(config)
    except ImportError as e:
        logger.error(f"Failed to import MCP server: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)
