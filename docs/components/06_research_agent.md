# Research Agent Component Specification

## Component Purpose

Provide intelligent research capabilities that go beyond simple chunk retrieval by using LLM analysis to answer complex questions about API documentation. While `searchAPI` returns raw documentation chunks, the Research Agent provides thoughtful analysis, code examples, and comprehensive answers using intelligent tool orchestration.

## Core Responsibilities

1. **Intelligent Search**: File-aware search with API context understanding
2. **Deep Schema Navigation**: Navigate 65+ level deep schemas when needed
3. **Iterative Retrieval**: LLM-driven multi-round information gathering
4. **Tool Orchestration**: Coordinate multiple async tool calls efficiently
5. **Answer Synthesis**: Generate comprehensive implementation guides
6. **Context Management**: Optimize token usage within LLM limits

## Architecture Overview

```
User Question + API Context
    ↓
LangGraph create_react_agent
    ↓
ReAct Loop: Reasoning + Acting
    ↓
Tool Calls: searchChunks & getChunks
    ↓
LLM Analysis & Synthesis
    ↓
Complete Answer with Examples
```

**Key Architecture Principles**:
- **Prebuilt ReAct Agent**: Use LangGraph's `create_react_agent` for proven patterns
- **Native Tool Calling**: Community packages handle tool binding automatically  
- **MessagesState**: Built-in conversation and state management
- **Minimal Code**: Single function call creates complete research workflow

## Enhanced Tool Set

### 1. Enhanced searchChunks Tool

**Purpose**: Intelligent search with file filtering and API context awareness

```python
async def searchChunks(
    vector_store: VectorStoreManager,
    api_context: str,
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False
) -> SearchResults:
    """
    Search for relevant chunks with optional file filtering.
    
    Args:
        vector_store: VectorStoreManager instance for search operations
        api_context: Available API files context string
        query: Natural language search query
        max_chunks: Maximum chunks to return (default: 25)
        file_filter: Optional file pattern (e.g., "sponsored-display", "dsp")
        include_references: Whether to include ref_ids in response
    
    Returns:
        SearchResults with chunk summaries and metadata
    """
```

**Enhanced Features**:
- **File Filtering**: Use existing metadata filters in ChromaDB
- **API Context**: Include available files in tool description
- **Smart Summaries**: Return chunk previews with ref_ids visible
- **Relevance Scoring**: Include similarity scores for LLM decision-making

**Data Structures**:
```python
@dataclass
class ChunkSummary:
    chunk_id: str
    title: str  # Human-readable title
    content_preview: str  # First 200 chars
    chunk_type: str  # "operation", "schema", "component"
    file_name: str  # Source file
    ref_ids: List[str]  # Available references
    relevance_score: float  # Search similarity score

@dataclass
class SearchResults:
    chunks: List[ChunkSummary]
    total_found: int
    search_time_ms: float
    files_searched: List[str]  # Which files were included
    api_context: str  # Available API files for reference
```

**Tool Description for LLM**:
```
searchChunks: Search API documentation chunks

Available API Files:
- sponsored-display-v3.json: Sponsored Display API (Campaigns, Targeting, Reports)
- amazon-dsp.json: DSP API (Audiences, Creative, Analytics) 
- seller-central.json: Seller API (Inventory, Orders, Reports)

Usage:
- searchChunks("campaign creation") - searches all files
- searchChunks("campaign creation", file_filter="sponsored-display") - only SD files
- Use file_filter when user mentions specific APIs or when you need focused results
```

### 2. New getChunks Tool

**Purpose**: Batch retrieval with intelligent depth control for deep schema navigation

```python
async def getChunks(
    vector_store: VectorStoreManager,
    chunk_ids: List[str],
    expand_depth: int = 3,
    max_total_chunks: int = 100
) -> RetrievalResults:
    """
    Retrieve multiple chunks by ID with controlled reference expansion.
    
    Args:
        vector_store: VectorStoreManager instance for retrieval operations
        chunk_ids: List of chunk IDs to retrieve
        expand_depth: Reference expansion depth (0-10+, default: 3)
        max_total_chunks: Limit total chunks including expansions (default: 100)
    
    Returns:
        RetrievalResults with full chunk content and expansion details
    """
```

**Deep Schema Navigation**:
- **Controlled Expansion**: 0 = just requested chunks, 5+ = deep schemas
- **Batch Efficiency**: Retrieve multiple chunks in single call
- **Expansion Tracking**: Show which chunks came from expansion
- **Token Management**: Respect total chunk limits

**Data Structures**:
```python
@dataclass
class FullChunk:
    chunk_id: str
    content: str  # Complete chunk content
    metadata: Dict[str, Any]
    source: str  # "requested" or "expanded"
    expansion_depth: int  # How deep this chunk was found
    ref_ids: List[str]  # References this chunk contains

@dataclass 
class RetrievalResults:
    requested_chunks: List[FullChunk]  # Originally requested
    expanded_chunks: List[FullChunk]   # From reference expansion
    total_chunks: int
    total_tokens: int
    expansion_stats: Dict[int, int]  # depth → chunk count
    truncated: bool  # Whether expansion was limited
```

**Depth Strategy Examples**:
- `expand_depth=0`: Just the requested chunks (for specific lookups)
- `expand_depth=1`: Include immediate references (basic context)
- `expand_depth=3`: Standard expansion (current searchAPI default)
- `expand_depth=5+`: Deep expansion (complex schema questions)

**Tool Description for LLM**:
```
getChunks: Retrieve specific chunks by ID with reference expansion

Use this when:
- You know specific chunk IDs from search results
- You need complete schemas (use expand_depth=3-5)
- You want to explore deeply nested structures (expand_depth=5-10)
- You need multiple related chunks efficiently

Examples:
- getChunks(["sd-api:CreateCampaign"], expand_depth=0) - just this schema
- getChunks(["sd-api:CreateCampaign"], expand_depth=5) - schema + all nested refs
- getChunks(["endpoint1", "endpoint2"], expand_depth=1) - two endpoints + immediate refs
```

### 3. API Context Enhancement

**Purpose**: Provide file awareness without additional tools

**System Prompt Enhancement**:
```python
RESEARCH_SYSTEM_PROMPT = """You are an expert API documentation researcher.

Available API Documentation:
{api_files_context}

Research Tools:
1. searchChunks(query, file_filter=None) - Search for relevant chunks
2. getChunks(chunk_ids, expand_depth=0) - Retrieve specific chunks with expansion

File Filtering Strategy:
- Use file_filter when user mentions specific APIs ("sponsored display" → "sponsored-display")
- Use file_filter to focus search when you identify relevant API from context
- Leave file_filter=None for broad searches across all APIs

Depth Control Strategy:
- expand_depth=0: Just specific chunks (quick lookups)
- expand_depth=1-2: Basic context (simple questions)  
- expand_depth=3-5: Complete schemas (implementation questions)
- expand_depth=5+: Deep nested structures (complex integration questions)

Remember: You can navigate 65+ levels deep when needed for complete examples."""
```

**API Context Generation**:
```python
def generate_api_context() -> str:
    """Generate API files context from api_index.json"""
    # Load existing api_index.json
    # Format as concise list with key info
    # Include file patterns for filtering
    return formatted_context
```

## LangGraph Implementation

### Prebuilt ReAct Agent Pattern

```python
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
async def search_chunks_tool(
    query: str,
    max_chunks: int = 25,
    file_filter: Optional[str] = None,
    include_references: bool = False
) -> Dict[str, Any]:
    """Search API documentation chunks with optional file filtering"""
    # Implementation using existing searchChunks function

@tool  
async def get_chunks_tool(
    chunk_ids: List[str], 
    expand_depth: int = 3,
    max_total_chunks: int = 100
) -> Dict[str, Any]:
    """Retrieve specific chunks with reference expansion"""
    # Implementation using existing getChunks function

def create_research_agent(llm_provider: str = "bedrock"):
    """Create Research Agent using LangGraph prebuilt patterns"""
    
    # Initialize LLM with community tool calling support
    if llm_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        model = ChatBedrockConverse(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            temperature=0.1,
            max_tokens=40000
        )
    elif llm_provider == "llama_cpp":
        from langchain_community.chat_models import ChatLlamaCpp
        model = ChatLlamaCpp(
            model_path="/path/to/model.gguf",
            temperature=0.1
        )
    else:
        model = init_chat_model(model=llm_provider)
    
    # Create ReAct agent - handles everything automatically
    return create_react_agent(
        model=model,
        tools=[search_chunks_tool, get_chunks_tool],
        prompt="""You are an expert API documentation researcher.

Available Tools:
- search_chunks_tool: Search for relevant documentation chunks
- get_chunks_tool: Retrieve specific chunks with deep reference expansion

Research Strategy:
1. Start with search_chunks_tool to find relevant chunks
2. Use file_filter when user mentions specific APIs
3. Use get_chunks_tool with appropriate expand_depth for detailed schemas
4. Provide comprehensive answers with working examples"""
    )

# Usage in MCP server
@mcp.tool()
async def research_api(question: str, llm_provider: str = "bedrock") -> str:
    """Research API documentation using ReAct agent"""
    agent = create_research_agent(llm_provider)
    
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": question}]
    })
    
    return result["messages"][-1].content
```

### Key Benefits of Prebuilt Approach

- ✅ **10x Less Code**: Single `create_react_agent()` call vs custom workflow
- ✅ **Proven Patterns**: Uses LangGraph's battle-tested ReAct implementation
- ✅ **Native Tool Calling**: Community packages handle tool binding automatically
- ✅ **Built-in Features**: Conversation management, streaming, error handling
- ✅ **MessagesState**: Automatic state management with message reducers
- ✅ **Easy Testing**: Simple `agent.invoke()` calls for testing

## Real-World Usage Examples

### Example 1: Deep Schema Navigation

**User Question**: "How do I create a sponsored display campaign with location targeting including all required fields?"

**ReAct Agent Flow**:
```python
# Single agent call handles entire research process
agent = create_research_agent("bedrock")

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": 
        "How do I create a sponsored display campaign with location targeting including all required fields?"
    }]
})

# Agent automatically:
# 1. Calls search_chunks_tool with file_filter="sponsored-display"
# 2. Analyzes results and identifies need for deep schema expansion
# 3. Calls get_chunks_tool with expand_depth=6 for nested location targeting
# 4. Synthesizes complete answer with working JSON example
```

### Example 2: Multi-API Integration

**User Question**: "How do I set up a campaign in Sponsored Display and then track it in DSP reports?"

**ReAct Agent Flow**:
```python
# Agent handles cross-API research automatically
agent = create_research_agent("bedrock")

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": 
        "How do I set up a campaign in Sponsored Display and then track it in DSP reports?"
    }]
})

# Agent automatically:
# 1. Searches across multiple APIs without file_filter
# 2. Identifies relevant endpoints in both sponsored-display and dsp
# 3. Retrieves detailed schemas for both CreateCampaign and DSP reporting
# 4. Synthesizes complete cross-API workflow with code examples
```

### Example 3: Error Handling Deep Dive

**User Question**: "What are all possible errors when creating campaigns and how do I handle them?"

**ReAct Agent Flow**:
```python
# Agent performs comprehensive error research
agent = create_research_agent("bedrock")

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": 
        "What are all possible errors when creating campaigns and how do I handle them?"
    }]
})

# Agent automatically:
# 1. Searches for error-related chunks with file_filter="sponsored-display"
# 2. Retrieves error schemas with deep expansion for nested error details
# 3. Gets endpoint-specific error responses and status codes
# 4. Generates comprehensive error handling guide with try/catch examples
```

## Implementation Benefits

### Handles Real Complexity
- **65-Level Schemas**: Can navigate arbitrarily deep with expand_depth
- **Large APIs**: File filtering prevents irrelevant results
- **Token Efficiency**: Batch retrieval reduces API calls
- **Smart Expansion**: Only expand when needed

### LLM-Driven Intelligence with ReAct  
- **Natural Tool Selection**: Agent decides which tools to use and when
- **Context-Aware Parameters**: Automatically chooses file_filter and expand_depth
- **Iterative Reasoning**: Built-in ReAct loop handles multi-step research
- **Source Integration**: Synthesizes information from multiple tool calls seamlessly

### Seamless Integration with LangGraph
- **Reuses Existing Infrastructure**: Wraps proven searchChunks/getChunks functions
- **Community Tool Calling**: Native support for Bedrock and LLaMA.cpp
- **Built-in Features**: Conversation history, streaming, error handling, persistence
- **LangSmith Integration**: Automatic tracing and debugging of agent behavior
- **Production Ready**: Battle-tested ReAct patterns with minimal custom code

## Configuration

### Environment Variables
```bash
# Research Agent Configuration (main agent for answering queries)
RESEARCH_AGENT_LLM_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0  # Main research agent model
RESEARCH_AGENT_LLM_MAX_TOKENS=40000                                   # Max tokens for comprehensive responses

# Re-ranker LLM Configuration (separate from main research agent)
RERANKER_LLM_MODEL=us.anthropic.claude-3-5-haiku-20241022-v1:0      # Fast re-ranking model
RERANKER_LLM_MAX_TOKENS=2048                                         # Sufficient for decision list output

# Tool Configuration  
GETCHUNKS_MAX_DEPTH=10                 # Maximum expansion depth
GETCHUNKS_MAX_TOTAL_CHUNKS=50         # Batch size limit
SEARCHCHUNKS_INCLUDE_API_CONTEXT=true # Include file context in responses

# LangGraph Configuration
LANGGRAPH_TRACING_ENABLED=true        # Enable LangSmith tracing
LANGGRAPH_CHECKPOINTER_ENABLED=false  # Enable conversation persistence
```

### MCP Server Integration

```python
@mcp.tool()
async def research_api(
    question: str,
    llm_provider: str = "bedrock"
) -> str:
    """
    Research API documentation using ReAct agent.
    
    Uses LangGraph's create_react_agent with community tool calling
    to provide intelligent research with searchChunks and getChunks tools.
    
    Args:
        question: Research question about API documentation
        llm_provider: "bedrock", "llama_cpp", or specific model string
    
    Returns:
        Comprehensive answer with examples and source references
    """
    agent = create_research_agent(llm_provider)
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content
```

## Testing Strategy

### Proof of Concept Tests
1. **AWS Bedrock Tool Calling**: Verify ChatBedrockConverse + tool binding works
2. **LLaMA.cpp Tool Calling**: Verify ChatLlamaCpp + forced tool choice works  
3. **Tool Integration**: Ensure searchChunks and getChunks work as LangChain tools
4. **Agent Creation**: Test create_react_agent with both LLM providers
5. **Real Data**: Use open-api-small-samples for realistic testing

### Real-World Test Cases
1. **Deep Schema Questions**: "Create campaign with all targeting options"
2. **Cross-File Integration**: "Set up campaign and configure reporting"  
3. **Error Handling**: "Handle all possible campaign creation errors"
4. **Performance**: "Complex questions under 10 seconds"
5. **Agent Reasoning**: Verify proper tool selection and parameter choices

### Success Criteria
- **Tool Calling Works**: Both AWS and LLaMA.cpp can call tools automatically
- **Agent Reasoning**: Proper ReAct loop with appropriate tool selection
- **Completeness**: >90% of complex questions answered fully  
- **Accuracy**: >95% of generated examples work correctly
- **Simplicity**: <100 lines of code for complete implementation

## Implementation Status

### ✅ Phase 1: Core Tools Implementation (COMPLETED)
**Status**: All core tools implemented and fully tested

- ✅ **searchChunks Tool**: Implemented with file filtering, reference inclusion, ChromaDB integration
  - Supports file_filter with substring matching via ChromaDB $in operator
  - Returns ChunkSummary objects with content previews and relevance scores
  - Proper timing and files_searched tracking
  - All tests passing: basic functionality, file filtering, references, non-matching filters

- ✅ **getChunks Tool**: Implemented with breadth-first reference expansion
  - Supports expand_depth from 0 (no expansion) to 10+ (deep schemas)
  - Implements BFS expansion with circular reference protection
  - Tracks expansion_stats by depth and handles max_total_chunks truncation
  - All tests passing: basic retrieval, reference expansion, circular protection, empty input

- ✅ **generate_api_context Tool**: Implemented with multiple format support
  - Handles both old and new api_index.json formats
  - Graceful error handling for missing files
  - All tests passing: real file and missing file scenarios

**Files Created**:
- `src/research_agent/tools.py` - Core tool implementations
- `src/research_agent/data_classes.py` - Data structures matching test expectations
- `src/research_agent/__init__.py` - Module exports

**Test Coverage**: 100% of research agent tool tests passing

### ✅ Phase 2: LangGraph Integration (COMPLETED)
**Status**: Production LangGraph agent implemented with Claude Sonnet 4

- ✅ **AWS Bedrock Tool Calling PoC**: Verified ChatBedrockConverse + tool binding works
- ✅ **Production LangGraph Agent**: Implemented with configurable model support
- ✅ **LangChain Tool Wrappers**: Converted searchChunks/getChunks to @tool decorators
- ✅ **MCP Server Integration**: research_api tool available in MCP server
- ✅ **Enhanced Prompt Engineering**: Comprehensive schema-based response generation
- ✅ **Configuration System**: Separate models for main agent (Sonnet 4) and re-ranker (Haiku)

### ⏳ Phase 3: Advanced Features (PENDING)
- LLaMA.cpp tool calling PoC (lower priority)
- LangSmith tracing setup
- Performance optimization
- Production deployment configuration

## Performance Improvements

**Recent Enhancements**: 
1. **Model Upgrade**: Upgraded main research agent from Haiku to Claude Sonnet 4 for better comprehension
2. **Token Limit Increase**: Increased max_tokens from 8192 to 40000 for comprehensive responses
3. **Enhanced Prompt Engineering**: Updated prompt to request complete schemas, examples, and hierarchical explanations
4. **Configuration Separation**: Separate models for main agent (comprehensive) and re-ranker (fast)

**Results**: 
- More comprehensive responses with complete schema definitions
- Better understanding of complex API relationships
- Detailed field-by-field explanations with constraints and examples
- Improved developer experience with implementable documentation

The research agent is now production-ready with comprehensive schema-based response generation.