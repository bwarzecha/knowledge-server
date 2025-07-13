# Two-Stage Query Expansion Architecture

## Overview

This document describes a **two-stage query expansion approach** that combines file selection via embeddings with context-rich API indexes. This approach allows us to capture comprehensive global metadata (title, description, version, tags) without being constrained by context limits.

## Architecture

### Stage 1: File Selection via Embeddings
```
User Query → File Embeddings Search → Select 3-5 Relevant API Files
```

### Stage 2: Query Expansion with Rich Context
```
Selected Files → Rich API Index → LLM Query Expansion → Enhanced Query
```

### Stage 3: Enhanced Retrieval
```
Enhanced Query → Vector Search (Filtered to Selected Files) → Operation Chunks
```

## Implementation Strategy

### 1. File-Level Embeddings

Create lightweight file embeddings for initial selection:

```python
# File embedding content
file_embedding_content = {
    'title': 'Sponsored Brands campaign management',
    'description': 'Create and manage Sponsored Brands campaigns...',
    'version': '4.0',
    'tags': ['Campaigns', 'Ad groups', 'Ads', 'Ad creatives'],
    'domains': ['advertising', 'campaigns', 'sponsored brands'],
    'key_paths': ['/campaigns', '/adGroups', '/ads'],
    'operations_summary': 'CRUD operations for campaigns, ad groups, and creatives'
}
```

### 2. Enhanced Context-Rich Index (No Size Limits)

Since we only need 3-5 files in context, we can include **comprehensive metadata**:

```python
# Enhanced index format (no compression needed)
enhanced_index = """
API: openapi.json
TITLE: Sponsored Brands campaign management
VERSION: 4.0
DESCRIPTION: Create and manage Sponsored Brands campaigns. Supports campaign creation, ad group management, and creative operations.
TAGS: Campaigns, Ad groups, Ads, Ad creatives, Recommendations
ENDPOINTS:
  POST /sb/v4/campaigns (Create campaigns) [campaignName, budget] {Campaigns}
  PUT /sb/v4/campaigns (Update campaigns) [campaignId, status] {Campaigns}
  POST /sb/v4/campaigns/list (List campaigns) [startIndex, count] {Campaigns}
  POST /sb/v4/campaigns/delete (Delete campaigns) [campaignIds] {Campaigns}
  POST /sb/v4/adGroups (Create ad groups) [campaignId, name] {Ad groups}
  PUT /sb/v4/adGroups (Update ad groups) [adGroupId, bid] {Ad groups}
  ...

API: openapi.yaml
TITLE: Amazon Ads API for Sponsored Display
VERSION: 3.0
DESCRIPTION: This API enables programmatic access for campaign creation, management, and reporting for Sponsored Display.
TAGS: Campaigns, Ad Groups, Product Ads, Targeting, Targeting Recommendations
ENDPOINTS:
  GET /sd/campaigns (Gets a list of campaigns) [portfolioIdFilter] {Campaigns}
  POST /sd/campaigns (Creates one or more campaigns) [name, tactic] {Campaigns}
  ...
"""
```

## Benefits of Two-Stage Approach

### ✅ **No Context Limit Constraints**
- **File selection**: Only 3-5 files needed in context
- **Rich metadata**: Can include full titles, descriptions, all tags
- **Comprehensive endpoints**: No need to limit endpoints per file
- **Detailed parameters**: Include all business-relevant parameters

### ✅ **Better Query Understanding**
- **Global context**: API purpose, version, business domain
- **Complete endpoint coverage**: All operations available for expansion
- **Rich parameter context**: Entity relationships and data flow
- **Tag-based categorization**: Business domain understanding

### ✅ **Improved Accuracy**
- **Relevant file focus**: Only expand using related APIs
- **Domain-specific expansion**: Context from similar business domains
- **Version awareness**: Handle API evolution and versioning
- **Tag-driven expansion**: Use business categorization

## Implementation Details

### Stage 1: File Selection System

```python
class FileSelector:
    """Select relevant API files using embeddings."""
    
    def __init__(self, vector_store_manager):
        self.vector_store = vector_store_manager
        self.file_embeddings = self._build_file_embeddings()
    
    def _build_file_embeddings(self):
        """Create embeddings for API file selection."""
        file_data = []
        
        for api_file in self.api_files:
            # Extract file-level metadata
            metadata = {
                'title': api_file.info.title,
                'description': api_file.info.description[:200],
                'version': api_file.info.version,
                'tags': [tag.name for tag in api_file.tags],
                'paths': list(api_file.paths.keys())[:10],
                'business_domains': self._extract_domains(api_file)
            }
            
            # Create searchable text
            searchable_text = f"""
            {metadata['title']} {metadata['description']}
            Tags: {' '.join(metadata['tags'])}
            Domains: {' '.join(metadata['business_domains'])}
            Paths: {' '.join(metadata['paths'])}
            """
            
            file_data.append({
                'file_id': api_file.name,
                'content': searchable_text,
                'metadata': metadata
            })
        
        return file_data
    
    def select_relevant_files(self, query: str, max_files: int = 5) -> List[str]:
        """Select most relevant API files for query expansion."""
        # Search file embeddings
        results = self.vector_store.search(
            query=query,
            collection_name="api_files",
            limit=max_files
        )
        
        return [result.metadata['file_id'] for result in results]
```

### Stage 2: Rich Context Generation

```python
class RichContextGenerator:
    """Generate comprehensive API context for selected files."""
    
    def generate_rich_context(self, selected_files: List[str]) -> str:
        """Generate detailed context from selected API files."""
        
        rich_context = []
        
        for file_id in selected_files:
            api_spec = self._load_api_spec(file_id)
            
            # Extract comprehensive metadata
            context = f"""
API: {file_id}
TITLE: {api_spec.info.title}
VERSION: {api_spec.info.version}
DESCRIPTION: {api_spec.info.description}
TAGS: {', '.join([tag.name for tag in api_spec.tags])}

ENDPOINTS:"""
            
            # Include ALL endpoints (no size constraints)
            for path, operations in api_spec.paths.items():
                for method, operation in operations.items():
                    endpoint_line = f"  {method.upper()} {path}"
                    
                    if operation.summary:
                        endpoint_line += f" ({operation.summary})"
                    
                    # Include all business parameters
                    params = self._extract_business_params(operation)
                    if params:
                        endpoint_line += f" [{', '.join(params)}]"
                    
                    # Include operation tags
                    if operation.tags:
                        endpoint_line += f" {{{', '.join(operation.tags)}}}"
                    
                    context += f"\n{endpoint_line}"
            
            rich_context.append(context)
        
        return "\n\n".join(rich_context)
```

### Stage 3: Enhanced Query Expansion

```python
class EnhancedQueryExpander:
    """Expand queries using rich context from selected files."""
    
    def expand_query(self, query: str, rich_context: str) -> str:
        """Expand query with comprehensive API context."""
        
        prompt = f"""You are an API documentation expert. Expand the user query with relevant technical terms from the selected API specifications.

SELECTED API CONTEXT:
{rich_context}

USER QUERY: {query}

Instructions:
1. Use the API titles, descriptions, and tags to understand business context
2. Add relevant HTTP methods and paths from the endpoints
3. Include version information if relevant
4. Add business domain terms from tags and descriptions
5. Include parameter names that show relevant entities
6. Focus on terms that will help find the correct API operations

EXPANDED QUERY:"""
        
        # Use LLM for expansion (with much richer context)
        expanded = self.llm.generate(prompt)
        
        return expanded
```

## Example Workflow

### Input Query
```
"How do I create a campaign?"
```

### Stage 1: File Selection
```python
# File embedding search finds:
selected_files = [
    "openapi.json",      # Sponsored Brands (campaigns)
    "openapi.yaml",      # Sponsored Display (campaigns)  
    "SponsoredProducts_prod_3p.json"  # Sponsored Products (campaigns)
]
```

### Stage 2: Rich Context Generation
```
API: openapi.json
TITLE: Sponsored Brands campaign management
VERSION: 4.0
DESCRIPTION: Create and manage Sponsored Brands campaigns
TAGS: Campaigns, Ad groups, Ads, Ad creatives
ENDPOINTS:
  POST /sb/v4/campaigns (Create campaigns) [campaignName, budget, tactic] {Campaigns}
  PUT /sb/v4/campaigns (Update campaigns) [campaignId, name, status] {Campaigns}
  ...

API: openapi.yaml  
TITLE: Amazon Ads API for Sponsored Display
VERSION: 3.0
DESCRIPTION: Programmatic access for campaign creation and management
TAGS: Campaigns, Ad Groups, Product Ads
ENDPOINTS:
  POST /sd/campaigns (Creates one or more campaigns) [name, tactic, budget] {Campaigns}
  GET /sd/campaigns (Gets a list of campaigns) [portfolioIdFilter] {Campaigns}
  ...
```

### Stage 3: Query Expansion
```
Original: "How do I create a campaign?"
Expanded: "create campaign POST /campaigns /sb/v4/campaigns /sd/campaigns sponsored brands display campaignName budget tactic advertising"
```

### Stage 4: Enhanced Retrieval
```python
# Enhanced query finds actual operation chunks:
results = vector_store.search(
    query=expanded_query,
    filters={"source_file": {"$in": selected_files}},  # Filter to selected files
    limit=10
)
# ✅ Now finds: CreateSponsoredBrandsCampaigns, POST /sd/campaigns operations
```

## Key Advantages

### 🎯 **Precision Through Selection**
- **Relevant context only**: No noise from unrelated APIs
- **Domain focus**: Business context from similar APIs
- **Efficient expansion**: Rich context without bloat

### 🚀 **Comprehensive Without Limits**
- **Full metadata**: Title, description, version, all tags
- **Complete endpoints**: All operations and parameters
- **Rich descriptions**: Business context and intent
- **Parameter context**: Entity relationships

### ✅ **Scalable Architecture**
- **File-level embeddings**: Fast selection from 1000+ APIs
- **Rich expansion context**: Detailed context for 3-5 files
- **No context limits**: Can include comprehensive metadata
- **Efficient filtering**: Subsequent search limited to relevant files

## Success Metrics

- **File Selection Accuracy**: % of relevant files in top 5
- **Query Expansion Quality**: Improvement in operation chunk retrieval
- **End-to-End Performance**: Overall scenario success rate
- **Context Efficiency**: Relevant information density in expansion

This two-stage approach enables **the best of both worlds**: scalable file selection and comprehensive query expansion without context constraints.