#!/usr/bin/env python3
"""Test script for MCP server functionality."""

import asyncio
from src.mcp_server.askAPI import askAPI

async def test_askAPI():
    """Test the askAPI function directly."""
    print("Testing askAPI function...")
    
    test_queries = [
        "how to create a user",
        "what endpoints are available",
        "how to authenticate"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await askAPI(query)
        print(f"Response: {result[:200]}...")
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_askAPI())