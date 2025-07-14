#!/usr/bin/env python3
"""
Comprehensive test scenarios for Knowledge Retriever using real sample files.

This script:
1. Indexes all sample OpenAPI files (3 large files with 1800+ schemas total)
2. Tests 8 realistic retrieval scenarios  
3. Validates that correct chunks and references are returned
4. Outputs detailed results to scenario_results.txt

USAGE: python test_scenarios.py

This will take 5-15 minutes to run as it processes large OpenAPI files and builds embeddings.
Results are saved to scenario_results.txt for analysis.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

from src.retriever import KnowledgeRetriever, RetrieverConfig
from src.vector_store.vector_store_manager import VectorStoreManager
from src.openapi_processor.processor import OpenAPIProcessor
from src.query_expansion.query_expander import QueryExpander


class ScenarioTester:
    """Test realistic Knowledge Retriever scenarios with sample data."""
    
    def __init__(self, enable_expansion: bool = False):
        """Initialize with temporary vector store."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = None
        self.retriever = None
        self.all_chunks = []
        self.output_file = "scenario_results.txt"
        self.start_time = time.time()
        self.enable_expansion = enable_expansion
        self.query_expander = None
        
    def setup(self):
        """Set up vector store and index sample files."""
        print("🔧 Setting up test environment...")
        
        # Create vector store with Arctic-Embed - optimized for retrieval
        self.vector_store = VectorStoreManager(
            persist_directory=str(Path(self.temp_dir) / "chromadb"),
            collection_name="scenario_test",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="mps",  # Fast 22M param model for testing
            max_tokens=8192,  # Much higher limit for comprehensive chunks
            reset_on_start=True
        )
        self.vector_store.setup()
        
        # Index all sample files
        processor = OpenAPIProcessor()
        samples_dir = "samples"
        
        print(f"📁 Processing files from {samples_dir}/...")
        chunks = processor.process_directory(samples_dir)
        self.all_chunks.extend(chunks)
        print(f"    Generated {len(chunks)} chunks from all sample files")
        
        print(f"🚀 Indexing {len(self.all_chunks)} total chunks...")
        # Use smaller batch size to avoid memory issues with Qwen3
        self.vector_store.add_chunks(self.all_chunks, batch_size=10)
        
        # Create retriever
        config = RetrieverConfig(
            max_primary_results=8,
            max_total_chunks=20,
            max_depth=3,
            token_limit=3000
        )
        self.retriever = KnowledgeRetriever(self.vector_store, config)
        
        # Set up query expander if enabled
        if self.enable_expansion:
            print("🔄 Setting up query expander...")
            self.query_expander = QueryExpander(
                llm_provider="bedrock", 
                model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"
            )
            if self.query_expander.llm and self.query_expander.llm.is_available():
                print("✅ Query expander ready!")
            else:
                print("⚠️ Query expander not available, will use original queries")
                self.query_expander = None
        
        print("✅ Setup complete!\n")
        
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def log(self, message: str):
        """Log message to both console and file."""
        print(message)
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def print_results(self, query: str, context, expected_terms: List[str] = None):
        """Print formatted results with validation."""
        self.log(f"🔍 Query: '{query}'")
        self.log(f"📊 Results: {len(context.primary_chunks)} primary + {len(context.referenced_chunks)} referenced chunks")
        self.log(f"⏱️  Time: {context.retrieval_stats.total_time_ms:.1f}ms")
        self.log(f"🔤 Tokens: {context.total_tokens}")
        
        if expected_terms:
            found_terms = set()
            all_text = " ".join([chunk.document.lower() for chunk in context.primary_chunks + context.referenced_chunks])
            
            for term in expected_terms:
                if term.lower() in all_text:
                    found_terms.add(term)
            
            self.log(f"✅ Found expected terms: {', '.join(found_terms)}")
            if len(found_terms) < len(expected_terms):
                missing = set(expected_terms) - found_terms
                self.log(f"❌ Missing terms: {', '.join(missing)}")
        
        # Show top chunks
        self.log("📋 Top primary chunks:")
        for i, chunk in enumerate(context.primary_chunks[:5], 1):
            chunk_type = chunk.metadata.get('type', 'unknown')
            source_file = chunk.metadata.get('source_file', 'unknown')
            self.log(f"  {i}. {chunk.id} ({chunk_type}) - score: {chunk.relevance_score:.3f} [{source_file}]")
            # Show snippet of document content
            doc_snippet = chunk.document[:200] + "..." if len(chunk.document) > 200 else chunk.document
            self.log(f"     Content: {doc_snippet}")
        
        if context.referenced_chunks:
            self.log("🔗 Referenced chunks:")
            for i, chunk in enumerate(context.referenced_chunks[:5], 1):
                chunk_type = chunk.metadata.get('type', 'unknown')
                source_file = chunk.metadata.get('source_file', 'unknown')
                self.log(f"  {i}. {chunk.id} ({chunk_type}) [{source_file}]")
        
        # Reference expansion stats
        self.log(f"🔍 Expansion stats: depth={context.retrieval_stats.depth_reached}, circular_refs={context.retrieval_stats.circular_refs_detected}")
        self.log("-" * 80)
    
    def retrieve_with_expansion(self, query: str):
        """Retrieve with optional query expansion."""
        original_query = query
        
        if self.enable_expansion and self.query_expander:
            try:
                expanded_query = self.query_expander.expand_query(query)
                self.log(f"🔍 Original query: {original_query}")
                self.log(f"🚀 Expanded query: {expanded_query}")
                query = expanded_query
            except Exception as e:
                self.log(f"⚠️ Query expansion failed: {e}, using original query")
                query = original_query
        
        return self.retriever.retrieve_knowledge(query)
        
    def scenario_1_campaign_operations(self):
        """Test: Find campaign management operations."""
        self.log("🎯 SCENARIO 1: Campaign Management Operations")
        query = "create campaign management operations"
        context = self.retrieve_with_expansion(query)
        
        expected_terms = ["campaign", "create", "POST", "update", "management"]
        self.print_results(query, context, expected_terms)
        
        # Validate we found campaign-related operations
        operation_chunks = [c for c in context.primary_chunks if c.metadata.get('type') == 'operation']
        self.log(f"✅ Validation: Found {len(operation_chunks)} operation chunks")
        
        campaign_related = any("campaign" in chunk.document.lower() or "campaign" in chunk.id.lower() 
                              for chunk in context.primary_chunks)
        self.log(f"✅ Validation: Campaign-related content found: {campaign_related}")
        
        return len(operation_chunks) > 0 and campaign_related
        
    def scenario_2_schema_references(self):
        """Test: Retrieve schema with its dependencies."""
        self.log("🎯 SCENARIO 2: Schema Definitions and References")
        query = "ad group schema structure definition"
        context = self.retriever.retrieve_knowledge(query)
        
        expected_terms = ["schema", "properties", "type", "adGroup", "definition"]
        self.print_results(query, context, expected_terms)
        
        # Should have both primary schema chunks and referenced dependencies
        schema_chunks = [c for c in context.primary_chunks + context.referenced_chunks 
                        if c.metadata.get('type') == 'component']
        self.log(f"✅ Validation: Found {len(schema_chunks)} schema/component chunks")
        return len(schema_chunks) > 0
        
    def scenario_3_error_handling(self):
        """Test: Find error handling and response codes."""
        self.log("🎯 SCENARIO 3: Error Handling and Response Codes")
        query = "error handling 400 bad request response codes"
        context = self.retriever.retrieve_knowledge(query)
        
        expected_terms = ["error", "400", "response", "bad", "request"]
        self.print_results(query, context, expected_terms)
        
        # Should find error-related content
        error_related = any(any(term in chunk.document.lower() for term in ["error", "400", "response"]) 
                           for chunk in context.primary_chunks)
        self.log(f"✅ Validation: Error-related content found: {error_related}")
        return error_related
        
    def scenario_4_authentication(self):
        """Test: Authentication and security requirements."""
        self.log("🎯 SCENARIO 4: Authentication and Security")
        query = "authentication security bearer token requirements"
        context = self.retriever.retrieve_knowledge(query)
        
        expected_terms = ["security", "authentication", "bearer", "token", "authorization"]
        self.print_results(query, context, expected_terms)
        
        auth_related = any(any(term in chunk.document.lower() for term in ["security", "auth", "token"]) 
                          for chunk in context.primary_chunks)
        self.log(f"✅ Validation: Authentication content found: {auth_related}")
        return auth_related
        
    def scenario_5_pagination_filtering(self):
        """Test: Complex query about pagination and filtering."""
        self.log("🎯 SCENARIO 5: Pagination and Filtering")
        query = "pagination limit offset filter parameters query"
        context = self.retriever.retrieve_knowledge(query)
        
        expected_terms = ["limit", "offset", "filter", "parameters", "pagination"]
        self.print_results(query, context, expected_terms)
        
        pagination_related = any(any(term in chunk.document.lower() for term in ["limit", "offset", "filter", "page"]) 
                               for chunk in context.primary_chunks)
        self.log(f"✅ Validation: Pagination content found: {pagination_related}")
        return pagination_related
        
    def scenario_6_specific_endpoint(self):
        """Test: Find specific endpoint and its requirements."""
        self.log("🎯 SCENARIO 6: Specific Endpoint Analysis")
        query = "sponsored products campaigns list GET endpoint"
        context = self.retriever.retrieve_knowledge(query)
        
        expected_terms = ["campaigns", "GET", "sponsored", "products", "list"]
        self.print_results(query, context, expected_terms)
        
        # Should find GET operations
        get_operations = [c for c in context.primary_chunks 
                         if c.metadata.get('method') == 'GET' or 'GET' in c.document]
        self.log(f"✅ Validation: Found {len(get_operations)} GET operations")
        
        sponsored_related = any("sponsored" in chunk.document.lower() or "campaign" in chunk.document.lower()
                              for chunk in context.primary_chunks)
        self.log(f"✅ Validation: Sponsored content found: {sponsored_related}")
        return len(get_operations) > 0 or sponsored_related
        
    def scenario_7_deep_references(self):
        """Test: Query that should trigger deep reference expansion."""
        self.log("🎯 SCENARIO 7: Deep Reference Chain")
        query = "campaign targeting criteria bid adjustments"
        context = self.retriever.retrieve_knowledge(query, max_depth=3, include_references=True)
        
        expected_terms = ["campaign", "targeting", "bid", "criteria", "adjustments"]
        self.print_results(query, context, expected_terms)
        
        # Should have expanded references if any exist
        has_references = any(chunk.metadata.get('ref_ids') for chunk in context.primary_chunks)
        expanded_refs = len(context.referenced_chunks) > 0
        
        self.log(f"✅ Validation: Has references in primary chunks: {has_references}")
        self.log(f"✅ Validation: Expanded references: {expanded_refs}")
        
        return len(context.primary_chunks) > 0
        
    def scenario_8_comparison_test(self):
        """Test: Compare results with and without reference expansion."""
        self.log("🎯 SCENARIO 8: Reference Expansion Comparison")
        query = "campaign budget settings configuration"
        
        context_no_refs = self.retriever.retrieve_knowledge(query, include_references=False)
        context_with_refs = self.retriever.retrieve_knowledge(query, include_references=True)
        
        self.log(f"Without references: {context_no_refs.total_chunks} chunks")
        self.log(f"With references: {context_with_refs.total_chunks} chunks")
        self.log(f"Reference expansion added: {context_with_refs.total_chunks - context_no_refs.total_chunks} chunks")
        
        # Should have same primary chunks
        same_primary = len(context_with_refs.primary_chunks) == len(context_no_refs.primary_chunks)
        no_refs_in_first = len(context_no_refs.referenced_chunks) == 0
        
        self.log(f"✅ Validation: Same primary chunks: {same_primary}")
        self.log(f"✅ Validation: No references when disabled: {no_refs_in_first}")
        
        return same_primary and no_refs_in_first
        
    def run_all_scenarios(self):
        """Run all test scenarios."""
        try:
            # Initialize output file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"KNOWLEDGE RETRIEVER COMPREHENSIVE SCENARIO TESTS\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
            
            self.setup()
            
            self.log("=" * 80)
            self.log("🧪 RUNNING KNOWLEDGE RETRIEVER SCENARIO TESTS")
            self.log(f"📁 Total chunks indexed: {len(self.all_chunks)}")
            self.log("=" * 80)
            
            results = {}
            results['campaign_ops'] = self.scenario_1_campaign_operations()
            results['schema_refs'] = self.scenario_2_schema_references()
            results['error_handling'] = self.scenario_3_error_handling()
            results['authentication'] = self.scenario_4_authentication()
            results['pagination'] = self.scenario_5_pagination_filtering()
            results['specific_endpoint'] = self.scenario_6_specific_endpoint()
            results['deep_refs'] = self.scenario_7_deep_references()
            results['comparison'] = self.scenario_8_comparison_test()
            
            # Summary
            self.log("=" * 80)
            self.log("📊 SCENARIO TEST SUMMARY")
            self.log("=" * 80)
            passed = sum(1 for result in results.values() if result)
            total = len(results)
            self.log(f"✅ Scenarios passed: {passed}/{total}")
            
            for scenario, result in results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                self.log(f"  {scenario}: {status}")
            
            elapsed = time.time() - self.start_time
            self.log(f"⏱️  Total test time: {elapsed:.1f} seconds")
            self.log("=" * 80)
            
            if passed == total:
                self.log("🎉 ALL SCENARIOS COMPLETED SUCCESSFULLY!")
            else:
                self.log(f"⚠️  {total - passed} scenarios failed - see details above")
            
            self.log(f"📝 Full results saved to: {self.output_file}")
            
        except Exception as e:
            error_msg = f"❌ Scenario test failed: {e}"
            self.log(error_msg)
            raise
        finally:
            self.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run knowledge retriever scenarios")
    parser.add_argument("--expand", action="store_true", 
                       help="Enable query expansion with Claude 3.5 Haiku")
    
    args = parser.parse_args()
    
    tester = ScenarioTester(enable_expansion=args.expand)
    tester.run_all_scenarios()