# Implementation Guide: Query Expansion with Context-Rich Index

## Overview

This guide provides step-by-step instructions for implementing query expansion using the context-rich API index approach in the Knowledge Server.

## Prerequisites

- Python 3.11+
- Virtual environment activated
- Required dependencies: `tiktoken`, `llama-cpp-python` (or similar local LLM)
- OpenAPI sample files for testing

## Implementation Steps

### Step 1: Create Context-Rich Index Generator

Create `src/query_expansion/context_rich_indexer.py`:

```python
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any

class ContextRichIndexer:
    """Generate context-rich API indexes for query expansion."""
    
    def generate_index(self, api_directories: List[str]) -> str:
        """Generate context-rich index from OpenAPI directories."""
        all_files = self._collect_openapi_files(api_directories)
        all_context_data = []
        
        for file_path in all_files:
            try:
                context_data = self._extract_context_from_file(file_path)
                if context_data:
                    all_context_data.append(context_data)
            except Exception as e:
                print(f"Warning: Skipping {file_path}: {e}")
        
        return self._format_flat_context_rich(all_context_data)
    
    def _collect_openapi_files(self, directories: List[str]) -> List[str]:
        """Collect all OpenAPI files from directories."""
        files = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                files.extend(dir_path.rglob("*.json"))
                files.extend(dir_path.rglob("*.yaml"))
                files.extend(dir_path.rglob("*.yml"))
        return [str(f) for f in files]
    
    def _extract_context_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract rich context from a single OpenAPI file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                spec = json.load(f)
            else:
                spec = yaml.safe_load(f)
        
        context_data = {
            'file': Path(file_path).stem,
            'endpoints': []
        }
        
        paths = spec.get('paths', {})
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoint = self._extract_endpoint_context(path, method, operation)
                    if endpoint:
                        context_data['endpoints'].append(endpoint)
        
        return context_data
    
    def _extract_endpoint_context(self, path: str, method: str, operation: Dict) -> Dict:
        """Extract context for a single endpoint."""
        # Extract business parameters (skip generic headers)
        params = []
        for param in operation.get('parameters', []):
            if isinstance(param, dict):
                name = param.get('name', '')
                if name and not self._is_generic_header(name):
                    params.append(name)
        
        # Extract path parameters
        path_params = re.findall(r'\\{([^}]+)\\}', path)
        params.extend(path_params)
        
        # Remove duplicates and limit
        params = list(set(params))[:3]
        
        return {
            'method': method.upper(),
            'path': path,
            'summary': operation.get('summary', ''),
            'parameters': params
        }
    
    def _is_generic_header(self, param_name: str) -> bool:
        """Check if parameter is a generic header to exclude."""
        generic_headers = [
            'authorization', 'content-type', 'accept', 'user-agent',
            'client-id', 'scope', 'amazon-advertising-api'
        ]
        return any(header in param_name.lower() for header in generic_headers)
    
    def _format_flat_context_rich(self, all_context_data: List[Dict]) -> str:
        """Format context data as flat, efficient index."""
        lines = []
        
        for data in all_context_data:
            file_name = data['file'][:12]  # Limit file name length
            endpoints = data['endpoints']
            
            endpoint_descriptions = []
            for endpoint in endpoints[:20]:  # Limit endpoints per file
                # Create: METHOD /path (summary) [params]
                desc = f"{endpoint['method']} {endpoint['path']}"
                
                # Add summary if meaningful
                summary = endpoint['summary']
                if summary and len(summary) > 8:
                    summary_short = summary[:30] + '...' if len(summary) > 30 else summary
                    desc += f" ({summary_short})"
                
                # Add key parameters
                params = [p for p in endpoint['parameters'] 
                         if any(term in p.lower() for term in ['id', 'campaign', 'ad', 'keyword'])]
                if params:
                    desc += f" [{','.join(params[:2])}]"
                
                endpoint_descriptions.append(desc)
            
            if endpoint_descriptions:
                lines.append(f"{file_name}: {' | '.join(endpoint_descriptions)}")
        
        return "\\n".join(lines)
```

### Step 2: Create Query Expansion Module

Create `src/query_expansion/query_expander.py`:

```python
import tiktoken
from typing import Dict, Any, Optional

class QueryExpander:
    """Expand user queries with API-specific terminology using local LLM."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.llm = self._initialize_llm(model_path)
    
    def _initialize_llm(self, model_path: Optional[str]):
        """Initialize local LLM (implement based on your choice)."""
        # Example with llama-cpp-python
        try:
            from llama_cpp import Llama
            if model_path:
                return Llama(model_path=model_path, n_ctx=8192)
        except ImportError:
            print("Warning: llama-cpp-python not available, using mock expansion")
        return None
    
    def expand_query(self, query: str, api_index: str) -> Dict[str, Any]:
        """Expand query with API-specific terms."""
        prompt = self._create_expansion_prompt(query, api_index)
        
        # Check token count
        prompt_tokens = len(self.tokenizer.encode(prompt))
        if prompt_tokens > 6000:  # Leave room for response
            return self._fallback_expansion(query, api_index)
        
        if self.llm:
            try:
                response = self.llm(prompt, max_tokens=200, temperature=0.3)
                expanded_query = self._extract_expansion(response['choices'][0]['text'])
                return {
                    'expanded_query': expanded_query,
                    'method': 'llm',
                    'original_query': query
                }
            except Exception as e:
                print(f"LLM expansion failed: {e}")
        
        # Fallback to rule-based expansion
        return self._fallback_expansion(query, api_index)
    
    def _create_expansion_prompt(self, query: str, api_index: str) -> str:
        """Create prompt for LLM expansion."""
        return f"""You are an API documentation expert. Expand user queries with relevant technical terms from the API index.

API INDEX:
{api_index}

USER QUERY: {query}

Instructions:
1. Keep the original user query intact
2. Add relevant HTTP methods (GET, POST, PUT, DELETE)
3. Add relevant path segments from the index
4. Add relevant technical terms
5. Focus on terms that will help find the right API operations

EXPANDED QUERY:"""
    
    def _fallback_expansion(self, query: str, api_index: str) -> Dict[str, Any]:
        """Rule-based fallback expansion."""
        query_lower = query.lower()
        index_lower = api_index.lower()
        
        expanded_terms = []
        
        # Add HTTP methods based on query intent
        if any(word in query_lower for word in ['create', 'add', 'new']):
            expanded_terms.append('POST')
        if any(word in query_lower for word in ['list', 'get', 'find', 'show']):
            expanded_terms.append('GET')
        if any(word in query_lower for word in ['update', 'modify', 'change']):
            expanded_terms.append('PUT')
        if any(word in query_lower for word in ['delete', 'remove', 'archive']):
            expanded_terms.append('DELETE')
        
        # Add relevant path segments
        if 'campaign' in query_lower and '/campaign' in index_lower:
            expanded_terms.append('/campaigns')
        if 'ad group' in query_lower or 'adgroup' in query_lower:
            if '/adgroup' in index_lower:
                expanded_terms.append('/adGroups')
        
        # Add business terms found in index
        business_terms = ['advertising', 'sponsored', 'targeting', 'keyword']
        for term in business_terms:
            if term in index_lower:
                expanded_terms.append(term)
        
        expanded_query = f"{query} {' '.join(expanded_terms)}" if expanded_terms else query
        
        return {
            'expanded_query': expanded_query,
            'method': 'fallback',
            'original_query': query
        }
    
    def _extract_expansion(self, llm_response: str) -> str:
        """Extract expanded query from LLM response."""
        # Clean up LLM response
        lines = llm_response.strip().split('\\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('USER', 'EXPANDED', 'Instructions')):
                return line
        return llm_response.strip()
```

### Step 3: Integrate with Test Scenarios

Modify `test_scenarios.py` to include query expansion:

```python
# Add to imports
from src.query_expansion.context_rich_indexer import ContextRichIndexer
from src.query_expansion.query_expander import QueryExpander

class ScenarioTester:
    def __init__(self):
        # ... existing initialization ...
        
        # Add query expansion components
        self.enable_expansion = False  # Control flag
        self.api_index = None
        self.query_expander = None
    
    def setup(self):
        # ... existing setup ...
        
        # Generate API index for query expansion
        if self.enable_expansion:
            print("🔄 Generating API index for query expansion...")
            indexer = ContextRichIndexer()
            self.api_index = indexer.generate_index(["samples"])
            self.query_expander = QueryExpander()
            print(f"✅ Generated API index ({len(self.api_index)} chars)")
    
    def retrieve_with_expansion(self, query: str, **kwargs):
        """Retrieve with optional query expansion."""
        original_query = query
        
        if self.enable_expansion and self.api_index and self.query_expander:
            expansion_result = self.query_expander.expand_query(query, self.api_index)
            expanded_query = expansion_result['expanded_query']
            
            print(f"🔍 Original: {original_query}")
            print(f"🚀 Expanded: {expanded_query}")
            
            # Use expanded query for retrieval
            query = expanded_query
        
        return self.retriever.retrieve_knowledge(query, **kwargs)
    
    def scenario_1_campaign_operations(self):
        """Updated scenario with expansion."""
        self.log("🎯 SCENARIO 1: Campaign Management Operations")
        query = "create campaign management operations"
        
        # Use expansion-aware retrieval
        context = self.retrieve_with_expansion(query)
        
        # ... rest of scenario logic ...
```

### Step 4: Command Line Interface

Add expansion control to test scenarios:

```python
# Add to main function in test_scenarios.py
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run knowledge retriever scenarios")
    parser.add_argument("--expand-queries", action="store_true", 
                       help="Enable query expansion with API index")
    parser.add_argument("--model-path", type=str,
                       help="Path to local LLM model for expansion")
    
    args = parser.parse_args()
    
    tester = ScenarioTester()
    tester.enable_expansion = args.expand_queries
    
    if args.model_path:
        tester.model_path = args.model_path
    
    tester.run_all_scenarios()
```

### Step 5: Testing and Validation

Run tests with and without expansion:

```bash
# Baseline test (current behavior)
python test_scenarios.py

# With query expansion
python test_scenarios.py --expand-queries

# With custom model
python test_scenarios.py --expand-queries --model-path /path/to/model.gguf
```

## Performance Optimization

### Index Size Management

1. **Monitor token usage**:
   ```python
   indexer = ContextRichIndexer()
   index = indexer.generate_index(directories)
   tokens = len(tiktoken.encoding_for_model("gpt-4").encode(index))
   print(f"Index tokens: {tokens} ({tokens/32000:.1%} of 32k limit)")
   ```

2. **Optimize for large API collections**:
   - Limit endpoints per file
   - Compress summaries
   - Filter out generic parameters
   - Group similar endpoints

### LLM Integration Options

1. **Local Models**:
   - llama-cpp-python with GGUF models
   - Ollama with REST API
   - transformers with local models

2. **Fallback Strategy**:
   - Always implement rule-based fallback
   - Test expansion quality before deployment
   - Monitor expansion success rates

## Troubleshooting

### Common Issues

1. **Index too large**: Reduce endpoints per file or compress summaries
2. **LLM not responding**: Check model path and memory requirements
3. **Poor expansion quality**: Improve prompt or use larger model
4. **Context limit exceeded**: Implement dynamic compression

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to query expansion
logger = logging.getLogger(__name__)
logger.debug(f"Expansion input: {query}")
logger.debug(f"Expansion output: {expanded_query}")
```

## Success Metrics

Track these metrics to validate effectiveness:

1. **Scenario Success Rate**: Target 8/8 instead of 7/8
2. **Expansion Quality**: Manual review of expanded queries
3. **Performance Impact**: Measure total query time
4. **Context Usage**: Monitor token consumption

## Next Steps

1. **Deploy to test environment**
2. **Run comparative analysis** (with/without expansion)
3. **Optimize based on results**
4. **Scale to full API collection**
5. **Implement production monitoring**

This implementation provides a robust foundation for query expansion that significantly improves the Knowledge Server's ability to understand user intent and find relevant API operations.