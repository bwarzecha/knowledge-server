# MCP Server Component Specification

## Component Purpose

Expose the clean `askAPI()` interface and handle LLM integration for question answering. This component provides the external interface that developers and LLMs interact with, orchestrating the complete knowledge retrieval and answer generation pipeline.

## Core Responsibilities

1. **MCP Server Setup**: Initialize and configure Model Context Protocol server
2. **API Interface**: Expose `askAPI(query)` function with proper MCP tool definition
3. **LLM Integration**: Support pluggable LLM providers (local models and AWS Bedrock)
4. **Context Management**: Assemble retrieved knowledge into LLM-optimized prompts
5. **Response Generation**: Generate comprehensive answers with examples and validation
6. **Error Handling**: Provide meaningful error responses for various failure modes
7. **Performance Monitoring**: Track response times and LLM usage

## Input/Output Contracts

### MCP Tool Definition
```python
# Tool exposed to LLM clients
{
    "name": "askAPI",
    "description": "Ask questions about API documentation and get comprehensive answers with examples",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about API usage, endpoints, schemas, or examples"
            }
        },
        "required": ["query"]
    }
}
```

### Response Format
```python
@dataclass
class APIResponse:
    answer: str                      # Comprehensive answer from LLM
    sources: List[str]               # Chunk IDs used for answer
    confidence: str                  # "high", "medium", "low" based on context quality
    context_stats: ContextStats      # Retrieval performance metrics
    llm_stats: LLMStats             # LLM usage and performance
    
@dataclass  
class ContextStats:
    total_chunks: int
    total_tokens: int
    retrieval_time_ms: float
    primary_results: int
    referenced_chunks: int

@dataclass
class LLMStats:
    provider: str                    # "local" or "aws_bedrock"
    model: str                       # Model identifier
    input_tokens: int                # Tokens sent to LLM
    output_tokens: int               # Tokens in response
    processing_time_ms: float        # LLM response time
    cost_estimate: Optional[float]   # Estimated cost (AWS only)
```

## Key Implementation Details

### MCP Server Configuration
**Standard Setup**: Use MCP SDK for Python with proper tool registration.

```python
from mcp import McpServer, Tool
from mcp.types import TextContent

class KnowledgeServerMCP:
    def __init__(self, config: Config):
        self.server = McpServer("knowledge-server")
        self.retriever = KnowledgeRetriever(...)
        self.llm_provider = self._initialize_llm_provider(config)
        self._register_tools()
    
    def _register_tools(self):
        """Register askAPI tool with MCP server"""
        
        @self.server.tool("askAPI")
        async def ask_api(query: str) -> str:
            """Ask questions about API documentation"""
            return await self._handle_api_query(query)
```

### Pluggable LLM Provider System
**Design Goal**: Support both local models and cloud APIs with identical interface.

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_answer(
        self, 
        query: str, 
        context: KnowledgeContext
    ) -> Tuple[str, LLMStats]:
        """Generate answer from query and context"""
        pass

class LocalLLMProvider(LLMProvider):
    def __init__(self, model_path: str, device: str = "mps"):
        # Initialize local model (e.g., Llama, Gemma)
        pass
        
    async def generate_answer(self, query: str, context: KnowledgeContext) -> Tuple[str, LLMStats]:
        # Use local model inference
        prompt = self._build_prompt(query, context)
        response = await self.model.generate(prompt)
        
        stats = LLMStats(
            provider="local",
            model=self.model_name,
            input_tokens=len(prompt.split()),  # Rough estimate
            output_tokens=len(response.split()),
            processing_time_ms=response.timing,
            cost_estimate=None
        )
        
        return response, stats

class AWSBedrockProvider(LLMProvider):
    def __init__(self, model_id: str, region: str = "us-east-1"):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        
    async def generate_answer(self, query: str, context: KnowledgeContext) -> Tuple[str, LLMStats]:
        # Use AWS Bedrock API
        prompt = self._build_prompt(query, context)
        
        request = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            })
        }
        
        start_time = time.time()
        response = await self.client.invoke_model(**request)
        processing_time = (time.time() - start_time) * 1000
        
        # Parse response and extract stats
        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]
        
        stats = LLMStats(
            provider="aws_bedrock",
            model=self.model_id,
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
            processing_time_ms=processing_time,
            cost_estimate=self._calculate_cost(result.get("usage", {}))
        )
        
        return answer, stats
```

### Prompt Engineering for API Documentation
**Critical Component**: Effective prompts ensure accurate, helpful responses.

```python
def _build_prompt(self, query: str, context: KnowledgeContext) -> str:
    """Build optimized prompt for API documentation questions"""
    
    # System prompt for API expertise
    system_prompt = """You are an expert API documentation assistant. Your job is to answer questions about API usage based ONLY on the provided documentation context.

Key guidelines:
1. Provide complete, accurate answers based on the context
2. Include working code examples when relevant
3. Mention specific endpoint paths, parameter names, and schema fields
4. If information is missing from context, say so explicitly
5. Structure responses clearly with headers and examples
6. Always validate examples against the provided schemas"""

    # Context assembly with chunk organization
    context_text = self._assemble_context_text(context)
    
    # User query with structured format
    user_prompt = f"""Context Documentation:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the documentation above. Include:
- Direct answer to the question
- Relevant code examples (if applicable)
- Required parameters and their types
- Expected response format
- Any important notes or limitations

Answer:"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"

def _assemble_context_text(self, context: KnowledgeContext) -> str:
    """Organize context chunks for optimal LLM understanding"""
    
    context_parts = []
    
    # Primary results first (most relevant)
    if context.primary_chunks:
        context_parts.append("## Primary Documentation (Most Relevant)")
        for i, chunk in enumerate(context.primary_chunks, 1):
            context_parts.append(f"### Result {i}: {chunk.id}")
            context_parts.append(chunk.document)
            context_parts.append("")  # Spacing
    
    # Referenced schemas second (dependencies)
    if context.referenced_chunks:
        context_parts.append("## Related Schemas and Dependencies")
        for chunk in context.referenced_chunks:
            context_parts.append(f"### Schema: {chunk.id}")
            context_parts.append(chunk.document)
            context_parts.append("")
    
    return "\n".join(context_parts)
```

### Response Quality Assessment
**Goal**: Provide confidence indicators to help users understand answer quality.

```python
def _assess_response_confidence(
    self, 
    query: str, 
    context: KnowledgeContext, 
    answer: str
) -> str:
    """Assess confidence in generated answer"""
    
    # High confidence criteria
    if (context.total_chunks >= 3 and 
        context.total_tokens >= 500 and
        len(context.primary_chunks) >= 2):
        return "high"
    
    # Medium confidence criteria  
    elif (context.total_chunks >= 2 and
          context.total_tokens >= 200):
        return "medium"
    
    # Low confidence (minimal context)
    else:
        return "low"
```

### Complete Query Handling Pipeline
**Main Flow**: Orchestrate retrieval, LLM generation, and response formatting.

```python
async def _handle_api_query(self, query: str) -> str:
    """Main query handling pipeline"""
    
    try:
        start_time = time.time()
        
        # Stage 1: Knowledge Retrieval
        logger.info(f"Processing query: {query}")
        context = await self.retriever.retrieve_knowledge(query)
        
        # Stage 2: Response Generation
        if context.total_chunks == 0:
            return self._handle_no_results(query)
            
        answer, llm_stats = await self.llm_provider.generate_answer(query, context)
        
        # Stage 3: Response Assembly
        confidence = self._assess_response_confidence(query, context, answer)
        
        response = APIResponse(
            answer=answer,
            sources=[chunk.id for chunk in context.primary_chunks + context.referenced_chunks],
            confidence=confidence,
            context_stats=ContextStats(
                total_chunks=context.total_chunks,
                total_tokens=context.total_tokens,
                retrieval_time_ms=context.retrieval_stats.total_time_ms,
                primary_results=len(context.primary_chunks),
                referenced_chunks=len(context.referenced_chunks)
            ),
            llm_stats=llm_stats
        )
        
        # Log performance metrics
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Query completed in {total_time:.1f}ms - Confidence: {confidence}")
        
        return self._format_response(response)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return self._handle_error(query, e)

def _format_response(self, response: APIResponse) -> str:
    """Format response for MCP client"""
    
    formatted = f"{response.answer}\n\n"
    
    # Add metadata footer
    formatted += f"---\n"
    formatted += f"Confidence: {response.confidence}\n"
    formatted += f"Sources: {len(response.sources)} chunks\n"
    formatted += f"Response time: {response.context_stats.retrieval_time_ms:.0f}ms retrieval + {response.llm_stats.processing_time_ms:.0f}ms generation\n"
    
    return formatted
```

## Configuration (.env Variables)

```bash
# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
MCP_SERVER_NAME=knowledge-server

# LLM Provider Configuration
LLM_PROVIDER=local                    # "local" or "aws_bedrock"

# Local LLM Configuration
LOCAL_MODEL_PATH=/path/to/model
LOCAL_MODEL_DEVICE=mps               # mps, cuda, or cpu
LOCAL_MAX_TOKENS=2000
LOCAL_TEMPERATURE=0.1

# AWS Bedrock Configuration  
AWS_REGION=us-east-1
AWS_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_MAX_TOKENS=2000
BEDROCK_TEMPERATURE=0.1

# Response Configuration
INCLUDE_PERFORMANCE_STATS=true       # Include timing in responses
INCLUDE_SOURCE_CHUNKS=true           # List chunk IDs used
CONFIDENCE_THRESHOLD_LOW=0.3         # Confidence assessment thresholds
CONFIDENCE_THRESHOLD_HIGH=0.7

# Error Handling
MAX_QUERY_LENGTH=500                 # Maximum query character limit
QUERY_TIMEOUT_SECONDS=30             # Total query timeout
ENABLE_GRACEFUL_DEGRADATION=true     # Partial results on timeout
```

## Definition of Done

### Functional Requirements
1. **MCP Server**: Properly initialized server exposing `askAPI` tool
2. **LLM Integration**: Support both local and AWS Bedrock providers
3. **Complete Pipeline**: Query → retrieval → LLM → formatted response
4. **Error Handling**: Graceful handling of all failure modes
5. **Performance Tracking**: Detailed metrics for optimization
6. **Response Quality**: Confidence assessment and source attribution

### Measurable Success Criteria
1. **Response Quality**: Generate coherent, accurate answers for sample API questions
2. **Performance**: <5 seconds total response time (retrieval + LLM)
3. **Reliability**: 100% uptime for valid queries (no crashes)
4. **LLM Integration**: Successfully work with both local and cloud providers
5. **Context Utilization**: Include source chunks and confidence levels in responses
6. **Error Recovery**: Meaningful error messages for all failure scenarios

### Integration Test Scenarios
1. **End-to-End**: Test complete workflow with sample queries and known good answers
2. **LLM Switching**: Verify both local and AWS Bedrock providers work correctly
3. **Context Quality**: Test with various context sizes and verify answer quality
4. **Error Scenarios**: Test with invalid queries, missing data, LLM failures
5. **Performance**: Measure response times under various loads
6. **MCP Compliance**: Verify tool registration and client integration work properly

## Implementation Guidelines

### Code Structure
```python
# Suggested file organization
mcp_server/
├── __init__.py
├── server.py              # Main MCP server setup and orchestration
├── llm_providers/          
│   ├── __init__.py
│   ├── base.py            # Abstract LLM provider interface
│   ├── local_provider.py  # Local model implementation
│   └── bedrock_provider.py # AWS Bedrock implementation
├── prompt_builder.py      # Prompt engineering and context assembly
└── response_formatter.py  # Response formatting and confidence assessment
```

### Key Classes
```python
class KnowledgeServerMCP:
    def __init__(self, config: Config):
        """Initialize MCP server with all dependencies"""
        
    async def start_server(self) -> None:
        """Start MCP server and register tools"""
        
    async def _handle_api_query(self, query: str) -> str:
        """Main query handling pipeline"""

class LLMProviderFactory:
    @staticmethod
    def create_provider(config: Config) -> LLMProvider:
        """Factory method for LLM provider creation"""

class PromptBuilder:
    def build_api_documentation_prompt(
        self, 
        query: str, 
        context: KnowledgeContext
    ) -> str:
        """Build optimized prompt for API questions"""
```

### Error Handling Strategy
- **Query Validation**: Check query length, format, and content
- **Retrieval Failures**: Handle empty results, timeout, database errors
- **LLM Failures**: Provider errors, timeout, quota exceeded
- **Response Formatting**: Handle malformed LLM responses
- **Graceful Degradation**: Partial responses when possible

### Performance Optimization
- **Async Processing**: Use async/await throughout pipeline
- **Connection Pooling**: Reuse database and API connections
- **Caching**: Cache frequent queries and provider responses
- **Timeout Management**: Respect timeout limits at each stage
- **Memory Management**: Monitor memory usage during processing

## Integration Points

### Upstream Dependencies
- **Knowledge Retriever**: Primary dependency for context retrieval
- **Configuration**: Uses .env settings for all behavioral control

### Downstream Dependencies
- **MCP Clients**: Serves requests from LLMs and development tools
- **Monitoring Systems**: Provides metrics and performance data

### External Integrations
```python
# MCP client integration example
from mcp import Client

client = Client("knowledge-server")
response = await client.call_tool("askAPI", {"query": "How do I create a campaign?"})
```

### Data Flow Validation
```python
def validate_api_response(response: APIResponse) -> bool:
    """Validate complete response structure and content"""
    # Check answer quality and completeness
    # Verify source attribution
    # Validate confidence assessment
    # Check performance metrics
```

## Testing Requirements

### Unit Tests
- Test LLM provider implementations with mock responses
- Test prompt building with various context types
- Test response formatting and confidence assessment
- Test error handling for various failure modes

### Integration Tests
- End-to-end query processing with real components
- LLM provider switching and configuration
- MCP server setup and tool registration
- Performance testing with concurrent requests

### Quality Assurance
- Manual testing with known API questions
- Validation of generated examples against schemas
- Comparison of local vs cloud LLM responses
- User experience testing with realistic queries

### Performance Tests
- Response time distribution under load
- Memory usage during concurrent processing
- LLM provider response time comparison
- Timeout handling and graceful degradation

This specification provides complete guidance for implementing the MCP Server component that exposes the clean `askAPI()` interface and integrates with both local and cloud LLM providers.