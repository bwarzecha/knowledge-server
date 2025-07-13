"""Simple logging setup - redirect all logs to stderr for MCP compatibility."""

import logging
import sys


def setup_logging(verbose: bool = False):
    """Setup logging to stderr. Use ERROR level for MCP, INFO for CLI."""
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr, force=True)
