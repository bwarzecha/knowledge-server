# Components Directory

This directory contains detailed specifications for each component of the Knowledge Server. Each specification serves as a comprehensive prompt for implementing that component, including all context, requirements, and design decisions from the prototype validation.

## Component Overview

### Core Components (Implement in Order)

1. **[OpenAPI Processor](./01_openapi_processor.md)** - Parse OpenAPI specs into chunks with metadata
2. **[Vector Store Manager](./02_vector_store_manager.md)** - ChromaDB operations and embedding management  
3. **[Knowledge Retriever](./03_knowledge_retriever.md)** - Semantic search with reference expansion
4. **[MCP Server](./04_mcp_server.md)** - `askAPI()` interface with LLM integration

### Supporting Components

5. **[Evaluation Framework](./05_evaluation_framework.md)** - Testing and quality measurement

## Implementation Guidelines

### Component Independence
Each component specification includes:
- **Clear Contracts**: Input/output data structures and interfaces
- **Configuration**: .env variables and settings
- **Definition of Done**: Measurable success criteria and test requirements
- **Integration Points**: How it connects to other components

### Development Approach
- Implement components in order (1-4, then 5)
- Test each component independently before integration
- Use sample OpenAPI specs in `/samples/` for testing
- Follow the tenets in ARCHITECTURE.md (simple, testable, short)

### Testing Strategy
- **Component Tests**: Each component tested with sample data
- **Integration Tests**: End-to-end workflows using real specs
- Focus on measurable outcomes, avoid gaming metrics

## Quick Start

1. Read [ARCHITECTURE.md](../ARCHITECTURE.md) for system overview
2. Set up development environment (Python 3.11+, venv, .env)
3. Start with OpenAPI Processor specification
4. Follow the Definition of Done for each component
5. Use Evaluation Framework to validate the complete system

Each component specification contains all the context needed to implement that component independently, drawn from the prototype learnings and validation results documented in `/docs/`.