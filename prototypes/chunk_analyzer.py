#!/usr/bin/env python3
"""
Chunk Analysis Script - Test if our chunks can answer real-world queries.
Simulates the multi-stage retrieval pipeline to validate chunk quality.
"""

import json
import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueryResult:
    query: str
    primary_chunks: List[Dict[str, Any]]
    referenced_chunks: List[Dict[str, Any]]
    total_tokens: int
    completeness_score: float
    missing_info: List[str]


class ChunkAnalyzer:
    def __init__(self, chunks_file: str):
        self.chunks = self._load_chunks(chunks_file)
        self.chunk_index = {chunk["id"]: chunk for chunk in self.chunks}
        print(f"📊 Loaded {len(self.chunks)} chunks for analysis")
    
    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(chunks_file, 'r') as f:
            return json.load(f)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
    
    def _semantic_search_simulation(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulate semantic search using keyword matching."""
        query_words = set(query.lower().split())
        
        # Add common variations and synonyms
        if "campaign" in query_words:
            query_words.update(["campaigns", "campaignid"])
        if "create" in query_words:
            query_words.update(["post", "add", "new"])
        if "get" in query_words or "list" in query_words:
            query_words.update(["retrieve", "fetch", "show"])
        if "error" in query_words:
            query_words.update(["exception", "failure", "problem"])
        if "field" in query_words:
            query_words.update(["property", "attribute", "parameter"])
        
        scored_chunks = []
        
        for chunk in self.chunks:
            score = 0
            content = (chunk["document"] + " " + str(chunk["metadata"])).lower()
            
            # Keyword matching with different weights
            for word in query_words:
                if word in content:
                    # Higher score for exact matches in operation names
                    if word in chunk["metadata"].get("operationId", "").lower():
                        score += 3
                    # Medium score for matches in document content
                    elif word in chunk["document"].lower():
                        score += 2
                    # Lower score for matches in metadata
                    else:
                        score += 1
            
            # Boost score for endpoint chunks vs schema chunks based on query
            if any(verb in query_words for verb in ["create", "get", "list", "update", "delete"]):
                if chunk["metadata"].get("type") == "endpoint":
                    score += 1
            elif any(term in query_words for term in ["field", "property", "schema", "object"]):
                if chunk["metadata"].get("type") == "schema":
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]
    
    def _expand_with_references(self, primary_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate Stage 2: Expand with referenced chunks."""
        referenced_chunks = []
        seen_ids = {chunk["id"] for chunk in primary_chunks}
        
        for chunk in primary_chunks:
            ref_ids = chunk["metadata"].get("ref_ids", [])
            for ref_id in ref_ids:
                if ref_id in self.chunk_index and ref_id not in seen_ids:
                    referenced_chunks.append(self.chunk_index[ref_id])
                    seen_ids.add(ref_id)
        
        return referenced_chunks
    
    def _assess_completeness(self, query: str, primary_chunks: List[Dict[str, Any]], 
                           referenced_chunks: List[Dict[str, Any]]) -> tuple[float, List[str]]:
        """Assess if the retrieved chunks can fully answer the query."""
        all_chunks = primary_chunks + referenced_chunks
        all_content = " ".join([chunk["document"] for chunk in all_chunks]).lower()
        
        missing_info = []
        completeness_score = 0.0
        
        query_lower = query.lower()
        
        # Check for different query patterns
        if "how do i" in query_lower or "how to" in query_lower:
            # Procedural queries need endpoint + schema info
            has_endpoint = any(chunk["metadata"].get("type") == "endpoint" for chunk in all_chunks)
            has_schema = any(chunk["metadata"].get("type") == "schema" for chunk in all_chunks)
            
            if has_endpoint:
                completeness_score += 0.6
            else:
                missing_info.append("Endpoint information")
                
            if has_schema:
                completeness_score += 0.4
            else:
                missing_info.append("Schema details")
        
        elif "what fields" in query_lower or "what properties" in query_lower:
            # Schema queries need detailed field information
            has_schema = any(chunk["metadata"].get("type") == "schema" for chunk in all_chunks)
            has_properties = "properties:" in all_content or "properties" in all_content
            
            if has_schema:
                completeness_score += 0.5
            else:
                missing_info.append("Schema definition")
                
            if has_properties:
                completeness_score += 0.5
            else:
                missing_info.append("Property details")
        
        elif "which apis" in query_lower or "what apis" in query_lower:
            # Cross-API queries need multiple API coverage
            api_names = set()
            for chunk in all_chunks:
                source_file = chunk["metadata"].get("source_file", "")
                if "sponsored" in source_file.lower():
                    if "display" in source_file.lower():
                        api_names.add("Sponsored Display")
                    elif "brands" in source_file.lower():
                        api_names.add("Sponsored Brands")
                    elif "products" in source_file.lower():
                        api_names.add("Sponsored Products")
            
            completeness_score = min(1.0, len(api_names) * 0.4)
            if len(api_names) < 2:
                missing_info.append("Cross-API coverage")
        
        elif "error" in query_lower:
            # Error queries need error schema or response codes
            has_error_info = any("error" in chunk["document"].lower() for chunk in all_chunks)
            has_response_codes = any(re.search(r'\b[45]\d{2}\b', chunk["document"]) for chunk in all_chunks)
            
            if has_error_info:
                completeness_score += 0.6
            else:
                missing_info.append("Error schema information")
                
            if has_response_codes:
                completeness_score += 0.4
            else:
                missing_info.append("HTTP error codes")
        
        else:
            # General queries - basic coverage
            if all_chunks:
                completeness_score = 0.8
            else:
                missing_info.append("No relevant chunks found")
                completeness_score = 0.0
        
        return min(1.0, completeness_score), missing_info
    
    def analyze_query(self, query: str) -> QueryResult:
        """Analyze a single query through the full retrieval pipeline."""
        print(f"\n🔍 Analyzing: \"{query}\"")
        
        # Stage 1: Semantic search
        primary_chunks = self._semantic_search_simulation(query)
        print(f"   Stage 1: Found {len(primary_chunks)} primary chunks")
        
        if primary_chunks:
            for chunk in primary_chunks[:2]:  # Show top 2
                print(f"      - {chunk['id']}")
        
        # Stage 2: Reference expansion
        referenced_chunks = self._expand_with_references(primary_chunks)
        print(f"   Stage 2: Added {len(referenced_chunks)} referenced chunks")
        
        # Stage 3: Completeness assessment
        completeness_score, missing_info = self._assess_completeness(query, primary_chunks, referenced_chunks)
        
        # Calculate total tokens
        all_content = " ".join([chunk["document"] for chunk in primary_chunks + referenced_chunks])
        total_tokens = self._estimate_tokens(all_content)
        
        print(f"   Result: {completeness_score:.1%} complete, {total_tokens} tokens")
        if missing_info:
            print(f"   Missing: {', '.join(missing_info)}")
        
        return QueryResult(
            query=query,
            primary_chunks=primary_chunks,
            referenced_chunks=referenced_chunks,
            total_tokens=total_tokens,
            completeness_score=completeness_score,
            missing_info=missing_info
        )
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run a comprehensive test suite of realistic queries."""
        test_queries = [
            # Endpoint discovery queries
            "How do I create a new sponsored display campaign?",
            "How do I get a list of campaigns?",
            "How to update a campaign?",
            "How do I delete an ad group?",
            
            # Schema inspection queries  
            "What fields are required in CreateCampaignRequest?",
            "What properties does the Campaign object have?",
            "What fields are in UpdateAttributesRequestContent?",
            "What are the types of creatives and their required fields?",
            
            # Cross-API queries
            "Which APIs support campaign frequency capping?",
            "What APIs support budget rules?",
            "Which APIs have optimization features?",
            
            # Error handling queries
            "What causes ACCESS_DENIED error in AdCatalog API?",
            "What are the possible error responses for campaign creation?",
            "What does error code 400 mean?",
            
            # Field-specific queries
            "What are valid values for campaign state?",
            "What is the format for targeting criteria?",
            "What are the required fields for creating an ad?",
            
            # Complex queries
            "How do I create a campaign with targeting and budget rules?",
            "What's the relationship between campaigns, ad groups and ads?",
            "How do I get reports for sponsored brands campaigns?"
        ]
        
        print("🧪 Running Test Suite")
        print("=" * 60)
        
        results = []
        total_score = 0
        
        for query in test_queries:
            result = self.analyze_query(query)
            results.append(result)
            total_score += result.completeness_score
        
        average_score = total_score / len(test_queries)
        
        print(f"\n📈 Test Suite Results")
        print("=" * 60)
        print(f"Average Completeness: {average_score:.1%}")
        print(f"Queries Tested: {len(test_queries)}")
        
        # Categorize results
        excellent = [r for r in results if r.completeness_score >= 0.9]
        good = [r for r in results if 0.7 <= r.completeness_score < 0.9]
        poor = [r for r in results if r.completeness_score < 0.7]
        
        print(f"Excellent (≥90%): {len(excellent)}")
        print(f"Good (70-89%): {len(good)}")
        print(f"Poor (<70%): {len(poor)}")
        
        if poor:
            print(f"\n❌ Queries needing improvement:")
            for result in poor:
                print(f"   {result.completeness_score:.1%} - {result.query}")
                if result.missing_info:
                    print(f"      Missing: {', '.join(result.missing_info)}")
        
        # Token usage analysis
        avg_tokens = sum(r.total_tokens for r in results) / len(results)
        max_tokens = max(r.total_tokens for r in results)
        print(f"\nToken Usage:")
        print(f"Average: {avg_tokens:.0f} tokens")
        print(f"Maximum: {max_tokens} tokens")
        
        return {
            "average_score": average_score,
            "results": results,
            "excellent": len(excellent),
            "good": len(good), 
            "poor": len(poor),
            "avg_tokens": avg_tokens,
            "max_tokens": max_tokens
        }


def main():
    chunks_file = "/Users/bartosz/dev/knowledge-server/chunks_export.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run the chunker first: python prototype_chunker.py")
        return
    
    analyzer = ChunkAnalyzer(chunks_file)
    summary = analyzer.run_test_suite()
    
    print(f"\n✅ Analysis complete!")
    print(f"Overall system readiness: {summary['average_score']:.1%}")
    
    if summary['average_score'] >= 0.8:
        print("🎉 Chunking strategy is ready for production!")
    elif summary['average_score'] >= 0.6:
        print("⚠️  Chunking strategy needs some improvements")
    else:
        print("❌ Chunking strategy needs significant work")


if __name__ == "__main__":
    main()