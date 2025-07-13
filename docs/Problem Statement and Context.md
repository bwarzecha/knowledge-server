# OpenAPI Knowledge Server - Problem Statement and Context

## Executive Summary

Modern AI-powered coding assistants face a critical limitation: they cannot effectively work with large OpenAPI specifications due to context window constraints. With enterprise API specs ranging from 50KB to over 2MB, containing hundreds of endpoints with deeply nested schema references, even the most advanced LLMs struggle to provide accurate, complete answers about API usage.

This project addresses this challenge by creating an MCP (Model Context Protocol) server that intelligently indexes OpenAPI specifications and exposes a simple interface for LLMs to query API knowledge with high accuracy and completeness.

## The Problem

### Scale and Complexity
- **200+ large OpenAPI specifications** stored locally
- Individual specs range from **50KB to 2MB+** 
- Hundreds of endpoints per spec with complex schema dependencies
- Deep reference chains (`$ref`) creating intricate webs of relationships
- Circular references and polymorphic schemas adding complexity

### Current Limitations
- LLM context windows (even at 200k tokens) cannot hold complete specs
- Naive chunking breaks schema references and loses critical relationships
- Simple search returns incomplete results, missing referenced schemas
- Developers and AI agents get partial, often incorrect API information

## Target Users

1. **AI Coding Assistants** - LLMs that need to generate API integration code
2. **Developers** - Engineers working with multiple complex APIs
3. **API Documentation Systems** - Tools that need to answer questions about APIs

## Key Use Cases

### 1. Endpoint Discovery
**Query**: "How do I create a new sponsored product campaign?"
**Need**: Complete endpoint details including method, path, parameters, request/response schemas, and examples

### 2. Schema Inspection  
**Query**: "What fields are required in UpdateAttributesRequestContent?"
**Need**: Full schema definition with all properties, types, validations, and inherited fields

### 3. Cross-API Search
**Query**: "Which APIs support campaign frequency capping?"
**Need**: Search across all specs to find relevant endpoints and schemas

### 4. Field Requirements
**Query**: "What are the types of creatives in Sponsored Brands and what are their required fields?"
**Need**: Detailed field information including enums, nested objects, and validation rules

### 5. Example Generation
**Query**: "Show me a complete example request for creating a campaign"
**Need**: Valid example that conforms to schema, with all required fields populated

## Success Criteria

### Accuracy Over Speed
- **>95% query coverage** - Retrieved chunks must contain all information needed to answer the query
- **Complete context retrieval** - All referenced schemas automatically included
- **Validated examples** - Generated examples must pass schema validation
- Response time in low seconds is acceptable (accuracy matters more than sub-second responses)

### Simplicity
- Clean, minimal API surface: `askAPI('your question here')`
- No complex configuration or setup required
- Works reliably with massive specs without manual intervention

### Intelligence
- Understands OpenAPI structure and relationships
- Automatically resolves and includes schema dependencies
- Provides contextual answers, not just raw spec fragments

## Constraints and Requirements

### Technical Constraints
- **Local deployment only** - Runs on developer's laptop
- **Python implementation** - Using MCP SDK for Python
- **Single user** - No concurrency requirements
- **ChromaDB** for vector storage (open to alternatives)

### Functional Requirements
- Handle both JSON and YAML OpenAPI formats
- Support OpenAPI 3.0+ specifications
- Work with specs containing 1000+ endpoints
- Resolve complex `$ref` chains (3-5 levels deep)
- Handle circular references gracefully
- Understand versioning and API evolution

## Why This Matters

This project enables a new paradigm for API documentation and usage:
- **AI agents can reliably generate correct API integration code**
- **Developers get complete, accurate answers about API usage**
- **Complex enterprise APIs become accessible to automation**
- **Knowledge about hundreds of APIs fits in a simple query interface**

The goal is to build a "beautiful small project which just works" - demonstrating that with intelligent chunking and retrieval strategies, we can make vast API knowledge accessible through a simple, elegant interface.