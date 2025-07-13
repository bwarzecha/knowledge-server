"""Research command for testing Research Agent functionality."""

import asyncio

from ...mcp_server.shared_resources import initialize_shared_resources
from ...research_agent import research_api_question
from ..config import Config


def research_command(config: Config, question: str, verbose: bool = False) -> None:
    """
    Research API documentation using the Research Agent.

    Args:
        config: Configuration object
        question: Research question to ask
        verbose: Whether to show detailed processing information
    """
    try:
        # Initialize shared resources
        if verbose:
            print("📚 Initializing shared resources...")
        initialize_shared_resources(config)
        if verbose:
            print("✅ Resources initialized")

        # Run research query
        if verbose:
            print(f"\n🔍 Researching: {question}")
            print("=" * 60)

        result = asyncio.run(research_api_question(question))

        if verbose:
            print("\n📝 Research Result:")
            print("-" * 60)

        print(result)

        if verbose:
            print("-" * 60)
            print("✅ Research completed successfully!")

    except Exception as e:
        print(f"❌ Research failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise
