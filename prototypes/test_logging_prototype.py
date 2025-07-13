#!/usr/bin/env python3
"""
Prototype to test global logger stderr redirect for MCP compatibility.
"""

import logging
import sys
import json


def setup_logging(level=logging.INFO):
    """Configure global logging to redirect all output to stderr."""
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stderr,  # Redirect ALL logging to stderr
        force=True  # Override any existing configuration
    )


def test_basic_logging():
    """Test basic logging behavior."""
    logger = logging.getLogger(__name__)
    
    print("=== Testing Basic Logging ===")
    print("This print() goes to stdout")
    
    logger.debug("This DEBUG goes to stderr")
    logger.info("This INFO goes to stderr") 
    logger.warning("This WARNING goes to stderr")
    logger.error("This ERROR goes to stderr")


def test_mcp_simulation():
    """Simulate MCP JSON protocol with mixed logging."""
    logger = logging.getLogger(__name__)
    
    print("=== Simulating MCP Protocol ===")
    
    # Simulate MCP server startup with logging
    logger.info("Starting MCP server...")
    logger.info("Loading vector store...")
    logger.info("Initializing components...")
    
    # Simulate clean JSON responses on stdout
    request1 = {"method": "tools/list", "id": 1}
    response1 = {
        "result": {
            "tools": [
                {"name": "ask_api", "description": "Ask API questions"}
            ]
        },
        "id": 1
    }
    
    logger.info(f"Received request: {request1['method']}")
    print(json.dumps(response1))  # Clean JSON to stdout
    
    # Another request/response
    request2 = {"method": "tools/call", "params": {"name": "ask_api", "arguments": {"query": "test"}}}
    response2 = {"result": {"content": [{"type": "text", "text": "API response here"}]}}
    
    logger.info(f"Processing tool call: {request2['params']['name']}")
    logger.info("Searching knowledge base...")
    print(json.dumps(response2))  # Clean JSON to stdout


def test_different_log_levels():
    """Test different logging levels."""
    print("=== Testing Different Log Levels ===")
    
    print("1. INFO level (normal operation):")
    setup_logging(logging.INFO)
    logger = logging.getLogger("test_info")
    logger.debug("This DEBUG should be hidden")
    logger.info("This INFO should show")
    logger.warning("This WARNING should show")
    logger.error("This ERROR should show")
    
    print("\n2. ERROR level (MCP silent mode):")
    setup_logging(logging.ERROR)
    logger = logging.getLogger("test_error")
    logger.debug("This DEBUG should be hidden")
    logger.info("This INFO should be hidden")
    logger.warning("This WARNING should be hidden") 
    logger.error("This ERROR should show")


def main():
    """Run all tests."""
    print("STDOUT: Starting logging prototype tests...")
    print("STDOUT: Check stderr for log messages")
    print()
    
    # Test 1: Basic logging setup
    setup_logging(logging.INFO)
    test_basic_logging()
    print()
    
    # Test 2: MCP simulation
    test_mcp_simulation()
    print()
    
    # Test 3: Different log levels
    test_different_log_levels()
    print()
    
    print("STDOUT: All tests completed!")
    print("STDOUT: If you see log messages above, they're going to stdout (BAD)")
    print("STDOUT: If you only see 'STDOUT:' messages above, logging works correctly!")


if __name__ == "__main__":
    main()