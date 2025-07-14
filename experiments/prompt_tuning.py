#!/usr/bin/env python3
"""
Prompt tuning experiments for LLM-based chunk filtering.
Tests multiple prompt variants to optimize filtering quality.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.cli.config import Config
from src.mcp_server.shared_resources import initialize_shared_resources
from src.research_agent.agent_tools import search_chunks_tool

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant to test."""
    name: str
    description: str
    prompt_function: callable


@dataclass
class TestResult:
    """Results from testing a prompt variant."""
    variant_name: str
    query: str
    original_count: int
    filtered_count: int
    reduction_ratio: float
    processing_time_ms: float
    fallback_used: bool
    decisions: Dict[str, str]
    relevance_score: float  # Manual assessment


def prompt_variant_1_instruction_based(query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int) -> str:
    """Instruction-based prompt with clear directives."""
    context_section = ""
    if search_context:
        context_section = f"\\nSearch Context: {search_context}\\n"
    
    prompt = f"""Analyze API documentation chunks for relevance. Filter and decide expansion strategy.

Query: "{query}"{context_section}

Instructions:
1. DISCARD chunks not directly relevant to the query
2. KEEP chunks that fully answer the query  
3. EXPAND_1/3/5 chunks needing context (1=basic, 3=moderate, 5=deep)
4. Target ~{target_count} relevant chunks

Chunks:
"""
    
    for chunk in chunk_info:
        prompt += f"""
ID: {chunk['id']}
Type: {chunk['type']} | File: {chunk['file']} | Score: {chunk['relevance_score']:.3f}
Content: {chunk['content'][:300]}...

"""
    
    prompt += "\\nRespond with: chunk_id -> action\\nExample: api:auth -> KEEP"
    return prompt


def prompt_variant_2_role_based(query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int) -> str:
    """Role-based prompt emphasizing expertise."""
    context_section = ""
    if search_context:
        context_section = f"\\nDeveloper Context: {search_context}\\n"
    
    prompt = f"""You are a senior API documentation expert helping developers find exactly what they need.

Developer Query: "{query}"{context_section}

Your task: Curate the most helpful chunks. Remove noise, keep essentials, expand what needs context.

Available chunks:
"""
    
    for chunk in chunk_info:
        prompt += f"""
🔍 {chunk['id']}
   📋 {chunk['type']} in {chunk['file']} 
   📊 Similarity: {chunk['relevance_score']:.1%}
   📄 {chunk['content'][:250]}...

"""
    
    prompt += f"""
As an expert, decide each chunk's fate:
- KEEP: Essential for answering the query
- EXPAND_1: Needs immediate context  
- EXPAND_3: Needs moderate expansion
- EXPAND_5: Needs deep context
- DISCARD: Not helpful for this query

Target ~{target_count} useful chunks. Format: chunk_id -> action
"""
    return prompt


def prompt_variant_3_criteria_focused(query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int) -> str:
    """Criteria-focused prompt with specific relevance guidelines."""
    context_section = ""
    if search_context:
        context_section = f"\\nContext: {search_context}\\n"
    
    prompt = f"""Filter API documentation chunks using these relevance criteria:

Query: "{query}"{context_section}

Relevance Criteria:
🎯 HIGH: Directly answers query, contains key terms, actionable content
📋 MEDIUM: Related concepts, supporting information, contextual details  
❌ LOW: Tangentially related, duplicate info, wrong API/version

Chunks to evaluate:
"""
    
    for chunk in chunk_info:
        prompt += f"""
{chunk['id']} | {chunk['type']} | Score: {chunk['relevance_score']:.3f}
{chunk['content'][:200]}...
---
"""
    
    prompt += f"""
Decisions (target {target_count} chunks):
- KEEP: High relevance, complete answer
- EXPAND_1/3/5: Medium relevance, needs context (depth 1-5)
- DISCARD: Low relevance

Format: chunk_id -> action
"""
    return prompt


def prompt_variant_4_concise(query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int) -> str:
    """Concise prompt for faster processing."""
    context_section = f" Context: {search_context}" if search_context else ""
    
    prompt = f"""Filter chunks for: "{query}"{context_section}

"""
    
    for chunk in chunk_info:
        prompt += f"{chunk['id']}: {chunk['content'][:150]}...\\n"
    
    prompt += f"""
Keep {target_count} most relevant. Actions: KEEP, EXPAND_1/3/5, DISCARD
Format: id -> action
"""
    return prompt


def prompt_variant_5_example_driven(query: str, search_context: Optional[str], chunk_info: List[Dict], target_count: int) -> str:
    """Example-driven prompt with few-shot learning."""
    context_section = ""
    if search_context:
        context_section = f"\\nContext: {search_context}\\n"
    
    prompt = f"""Filter API documentation chunks for relevance. Learn from examples:

Query: "authentication methods"
- api:auth:bearer -> KEEP (directly about authentication)
- api:auth:errors -> EXPAND_1 (auth-related errors, needs context)  
- api:campaigns:create -> DISCARD (not about authentication)

Query: "{query}"{context_section}

Chunks:
"""
    
    for chunk in chunk_info:
        prompt += f"""
{chunk['id']}: {chunk['content'][:200]}...
"""
    
    prompt += f"""
Apply same logic. Target {target_count} chunks.
Actions: KEEP (direct answer), EXPAND_1/3/5 (needs context), DISCARD (irrelevant)
Format: chunk_id -> action
"""
    return prompt


async def test_prompt_variant(variant: PromptVariant, test_queries: List[Dict]) -> List[TestResult]:
    """Test a prompt variant against multiple queries."""
    results = []
    
    # Temporarily replace the prompt function
    from src.research_agent import tools
    original_prompt_func = tools._build_intelligence_prompt
    tools._build_intelligence_prompt = variant.prompt_function
    
    try:
        for query_data in test_queries:
            print(f"  Testing: {query_data['query'][:40]}...")
            
            try:
                result = await search_chunks_tool.ainvoke({
                    "query": query_data["query"],
                    "max_chunks": query_data.get("max_chunks", 4),
                    "rerank": True,
                    "search_context": query_data.get("context")
                })
                
                if result.get('filtering_stats'):
                    stats = result['filtering_stats']
                    
                    # Simple relevance scoring based on expected terms
                    relevance_score = 0.0
                    if 'expected_terms' in query_data:
                        total_chunks = len(result['chunks'])
                        relevant_chunks = 0
                        for chunk in result['chunks']:
                            chunk_text = (chunk['title'] + ' ' + chunk['chunk_id']).lower()
                            if any(term.lower() in chunk_text for term in query_data['expected_terms']):
                                relevant_chunks += 1
                        relevance_score = relevant_chunks / total_chunks if total_chunks > 0 else 0.0
                    
                    test_result = TestResult(
                        variant_name=variant.name,
                        query=query_data["query"],
                        original_count=stats['original_count'],
                        filtered_count=stats['filtered_count'],
                        reduction_ratio=stats['reduction_ratio'],
                        processing_time_ms=stats['processing_time_ms'],
                        fallback_used=stats['fallback_used'],
                        decisions=stats.get('decisions', {}),
                        relevance_score=relevance_score
                    )
                    results.append(test_result)
                else:
                    print(f"    ❌ No filtering stats for {query_data['query']}")
                    
            except Exception as e:
                print(f"    ❌ Error testing {query_data['query']}: {e}")
                
    finally:
        # Restore original prompt function
        tools._build_intelligence_prompt = original_prompt_func
    
    return results


async def run_prompt_tuning_experiments():
    """Run comprehensive prompt tuning experiments."""
    print("🧪 === Prompt Tuning Experiments ===\\n")
    
    # Initialize
    config = Config()
    initialize_shared_resources(config)
    
    # Define test queries
    test_queries = [
        {
            "query": "authentication bearer token headers",
            "context": "Developer needs to authenticate API calls with proper headers",
            "expected_terms": ["auth", "bearer", "token", "header"],
            "max_chunks": 4
        },
        {
            "query": "create campaign required fields",
            "context": "Developer wants to create campaigns and needs all required fields",
            "expected_terms": ["campaign", "create", "required", "field"],
            "max_chunks": 4
        },
        {
            "query": "error handling status codes",
            "context": "Developer needs to handle API errors and understand status codes",
            "expected_terms": ["error", "status", "code", "exception"],
            "max_chunks": 5
        },
        {
            "query": "campaign optimization rules",
            "context": "Developer wants to set up automatic bid optimization for campaigns",
            "expected_terms": ["optimization", "rules", "bid", "campaign"],
            "max_chunks": 3
        }
    ]
    
    # Define prompt variants
    variants = [
        PromptVariant("Instruction", "Clear step-by-step instructions", prompt_variant_1_instruction_based),
        PromptVariant("Role-Based", "Expert persona with developer focus", prompt_variant_2_role_based),
        PromptVariant("Criteria", "Specific relevance criteria", prompt_variant_3_criteria_focused),
        PromptVariant("Concise", "Minimal prompt for speed", prompt_variant_4_concise),
        PromptVariant("Examples", "Few-shot learning with examples", prompt_variant_5_example_driven),
    ]
    
    # Test each variant
    all_results = []
    for variant in variants:
        print(f"\\n🧪 Testing: {variant.name} - {variant.description}")
        results = await test_prompt_variant(variant, test_queries)
        all_results.extend(results)
        
        if results:
            avg_reduction = sum(r.reduction_ratio for r in results) / len(results)
            avg_time = sum(r.processing_time_ms for r in results) / len(results)
            avg_relevance = sum(r.relevance_score for r in results) / len(results)
            fallback_rate = sum(1 for r in results if r.fallback_used) / len(results)
            
            print(f"  📊 Avg Reduction: {avg_reduction:.1%}")
            print(f"  ⏱️  Avg Time: {avg_time:.0f}ms")
            print(f"  🎯 Avg Relevance: {avg_relevance:.1%}")
            print(f"  🔄 Fallback Rate: {fallback_rate:.1%}")
    
    # Analyze results
    print("\\n📊 === Experiment Results Summary ===")
    
    # Group by variant
    variant_results = {}
    for result in all_results:
        if result.variant_name not in variant_results:
            variant_results[result.variant_name] = []
        variant_results[result.variant_name].append(result)
    
    # Calculate metrics per variant
    print("\\n| Variant | Avg Reduction | Avg Time (ms) | Avg Relevance | Fallback Rate |")
    print("|---------|---------------|---------------|---------------|---------------|")
    
    best_variant = None
    best_score = -1
    
    for variant_name, results in variant_results.items():
        if not results:
            continue
            
        avg_reduction = sum(r.reduction_ratio for r in results) / len(results)
        avg_time = sum(r.processing_time_ms for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        fallback_rate = sum(1 for r in results if r.fallback_used) / len(results)
        
        # Composite score (relevance most important, then reduction, then speed)
        score = (avg_relevance * 0.5) + (avg_reduction * 0.3) + ((10000 - avg_time) / 10000 * 0.2) - (fallback_rate * 0.5)
        
        print(f"| {variant_name:8} | {avg_reduction:12.1%} | {avg_time:12.0f} | {avg_relevance:12.1%} | {fallback_rate:12.1%} |")
        
        if score > best_score:
            best_score = score
            best_variant = variant_name
    
    print(f"\\n🏆 **Best Variant: {best_variant}** (Score: {best_score:.3f})")
    
    # Save detailed results
    results_path = Path("experiments/prompt_tuning_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    results_data = {
        "timestamp": time.time(),
        "best_variant": best_variant,
        "best_score": best_score,
        "detailed_results": [
            {
                "variant": r.variant_name,
                "query": r.query,
                "original_count": r.original_count,
                "filtered_count": r.filtered_count,
                "reduction_ratio": r.reduction_ratio,
                "processing_time_ms": r.processing_time_ms,
                "relevance_score": r.relevance_score,
                "fallback_used": r.fallback_used
            }
            for r in all_results
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\\n💾 Detailed results saved to: {results_path}")
    print("\\n✅ Prompt tuning experiments completed!")


if __name__ == "__main__":
    asyncio.run(run_prompt_tuning_experiments())