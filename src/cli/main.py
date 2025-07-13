"""Main CLI entry point for knowledge server."""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.commands.ask import ask_command
from src.cli.commands.index import index_command
from src.cli.commands.research import research_command
from src.cli.commands.serve import serve_command
from src.cli.config import Config
from src.utils.logging_config import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="knowledge-server",
        description="Knowledge Server - MCP server for API documentation",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Build vector store and API index from OpenAPI specs and markdown files"
    )
    index_parser.add_argument("--config", help="Path to .env configuration file", default=None)
    index_parser.add_argument("--skip-openapi", action="store_true", help="Skip processing OpenAPI specifications")
    index_parser.add_argument("--skip-markdown", action="store_true", help="Skip processing markdown files")
    index_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens per markdown chunk (default: 1000, max: 8000)"
    )
    index_parser.add_argument(
        "--markdown-dir", help="Directory containing markdown files (default: same as OpenAPI specs directory)"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start MCP server using pre-built indices")
    serve_parser.add_argument("--config", help="Path to .env configuration file", default=None)
    serve_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show startup messages (default: silent for MCP compatibility)",
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask questions about API documentation (test askAPI functionality)")
    ask_parser.add_argument("query", help="Question to ask about the API documentation")
    ask_parser.add_argument(
        "--max-response-length",
        type=int,
        default=4000,
        help="Maximum response length in tokens (default: 4000)",
    )
    ask_parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Maximum chunks to retrieve (default: 50)",
    )
    ask_parser.add_argument(
        "--no-references",
        action="store_true",
        help="Disable reference following for faster responses",
    )
    ask_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum reference expansion depth (default: 3)",
    )
    ask_parser.add_argument("--config", help="Path to .env configuration file", default=None)
    ask_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    # Research command
    research_parser = subparsers.add_parser("research", help="Research questions using intelligent ReAct agent")
    research_parser.add_argument("question", help="Research question about API documentation")
    research_parser.add_argument("--config", help="Path to .env configuration file", default=None)
    research_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging based on command and verbose flag
    verbose = getattr(args, "verbose", False)
    # Index command shows progress by default (it's a long-running operation)
    if args.command == "index":
        setup_logging(verbose=True)
    else:
        setup_logging(verbose=verbose)

    # Load configuration
    config = Config(args.config)

    # Execute command
    if args.command == "index":
        index_command(
            config=config,
            skip_openapi=args.skip_openapi,
            skip_markdown=args.skip_markdown,
            max_tokens=args.max_tokens,
            markdown_dir=args.markdown_dir,
        )
    elif args.command == "serve":
        serve_command(config, verbose=verbose)
    elif args.command == "ask":
        ask_command(
            config=config,
            query=args.query,
            max_response_length=args.max_response_length,
            max_chunks=args.max_chunks,
            include_references=not args.no_references,
            max_depth=args.max_depth,
            verbose=verbose,
        )
    elif args.command == "research":
        research_command(
            config=config,
            question=args.question,
            verbose=verbose,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
