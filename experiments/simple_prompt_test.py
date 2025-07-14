#!/usr/bin/env python3
"""
Simple prompt testing - test current prompt vs one optimized variant.
Focus on ensuring parsing works and measuring real improvements.
"""

import asyncio
import json
import time
from pathlib import Path

from src.cli.config import Config
from src.mcp_server.shared_resources import initialize_shared_resources
from src.research_agent.agent_tools import search_chunks_tool


def optimized_prompt(query: str, search_context, chunk_info, target_count: int) -> str:
    """Optimized prompt - more focused and explicit about output format."""
    
    context_section = ""
    if search_context:
        context_section = f"""
SEARCH GOAL: {search_context}
Use this context to make smarter relevance decisions.
"""
    
    prompt = f"""You are filtering API documentation chunks for developer queries.

QUERY: "{query}"{context_section}

CHUNKS TO EVALUATE:
"""
    
    for chunk in chunk_info:
        prompt += f"""
ID: {chunk['id']}
TYPE: {chunk['type']} | FILE: {chunk['file']} | RELEVANCE: {chunk['relevance_score']:.3f}
CONTENT: {chunk['content'][:400]}

---
"""
    
    prompt += f"""
FILTERING INSTRUCTIONS:
1. Analyze each chunk's relevance to the query and search goal
2. Target keeping ~{target_count} most useful chunks
3. Choose action for each relevant chunk:
   - KEEP: Chunk directly answers the query
   - EXPAND_1: Needs basic context (1 level)
   - EXPAND_3: Needs moderate context (3 levels) 
   - EXPAND_5: Needs deep context (5 levels)
   - DISCARD: Not relevant (don't include in response)

REQUIRED OUTPUT FORMAT:
One line per chunk you want to keep. Use exact chunk ID followed by arrow and action.
chunk_id -> action

EXAMPLE:
openapi.yaml:components/securitySchemes/bearerAuth -> KEEP
api:campaigns:create -> EXPAND_3

IMPORTANT: 
- Use exact chunk IDs shown above
- Only include chunks you want to keep (omit DISCARD chunks)
- Use -> arrow format exactly as shown
"""
    
    return prompt


async def test_current_vs_optimized():
    """Test current prompt vs optimized prompt."""
    print("=== Testing Current vs Optimized Prompt ===\n")
    
    # Initialize
    config = Config()
    initialize_shared_resources(config)
    
    # Test queries
    test_queries = [
        {
            "query": "authentication API key bearer token",
            "context": "Developer needs to authenticate API calls with proper authorization headers",
        },
        {
            "query": "create campaign required fields",
            "context": "Developer wants to create advertising campaigns and needs all required parameters",
        },
        {
            "query": "error handling status codes",
            "context": "Developer needs to handle API errors and understand response codes",
        }
    ]
    
    results = {
        "timestamp": time.time(),
        "tests": []
    }
    
    for i, query_data in enumerate(test_queries, 1):
        print(f"Test {i}: {query_data['query']}")
        
        # Test current prompt
        print("  Testing current prompt...")
        try:
            current_result = await search_chunks_tool.ainvoke({
                "query": query_data["query"],
                "max_chunks": 4,
                "rerank": True,
                "search_context": query_data["context"]
            })
            
            current_stats = current_result.get('filtering_stats', {})
            print(f"    Current: {current_stats.get('original_count', 0)} -> {current_stats.get('filtered_count', 0)} chunks")
            print(f"    Reduction: {current_stats.get('reduction_ratio', 0):.1%}")
            print(f"    Time: {current_stats.get('processing_time_ms', 0):.0f}ms")
            print(f"    Fallback: {current_stats.get('fallback_used', True)}")
            
        except Exception as e:
            print(f"    ❌ Current failed: {e}")
            current_stats = {"error": str(e)}
        
        # Test optimized prompt
        print("  Testing optimized prompt...")
        try:
            # Temporarily replace prompt function
            from src.research_agent import tools
            original_prompt = tools._build_intelligence_prompt
            tools._build_intelligence_prompt = optimized_prompt
            
            optimized_result = await search_chunks_tool.ainvoke({
                "query": query_data["query"],
                "max_chunks": 4,
                "rerank": True,
                "search_context": query_data["context"]
            })
            
            # Restore original prompt
            tools._build_intelligence_prompt = original_prompt
            
            optimized_stats = optimized_result.get('filtering_stats', {})
            print(f"    Optimized: {optimized_stats.get('original_count', 0)} -> {optimized_stats.get('filtered_count', 0)} chunks")
            print(f"    Reduction: {optimized_stats.get('reduction_ratio', 0):.1%}")
            print(f"    Time: {optimized_stats.get('processing_time_ms', 0):.0f}ms")
            print(f"    Fallback: {optimized_stats.get('fallback_used', True)}")
            
        except Exception as e:
            print(f"    ❌ Optimized failed: {e}")
            optimized_stats = {"error": str(e)}
            # Restore original prompt on error
            from src.research_agent import tools
            tools._build_intelligence_prompt = original_prompt
        
        # Store results
        test_result = {
            "query": query_data["query"],
            "context": query_data["context"],
            "current": current_stats,
            "optimized": optimized_stats
        }
        results["tests"].append(test_result)
        print()
    
    # Save results
    results_file = Path("experiments/prompt_comparison_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Summary
    print("\n=== Summary ===")
    current_successes = sum(1 for test in results["tests"] if not test["current"].get("fallback_used", True))
    optimized_successes = sum(1 for test in results["tests"] if not test["optimized"].get("fallback_used", True))
    
    print(f"Current prompt: {current_successes}/{len(test_queries)} successful")
    print(f"Optimized prompt: {optimized_successes}/{len(test_queries)} successful")
    
    if optimized_successes > current_successes:
        print("✅ Optimized prompt performs better!")
    elif current_successes > optimized_successes:
        print("✅ Current prompt is already optimal")
    else:
        print("🤔 Results are mixed - need more testing")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_current_vs_optimized())