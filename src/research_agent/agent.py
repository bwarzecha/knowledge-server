"""Research Agent using LangGraph's create_react_agent."""

import logging

from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

from .agent_tools import get_chunks_tool, search_chunks_tool
from .tools import generate_api_context

logger = logging.getLogger(__name__)


def create_research_agent():
    """Create Research Agent using LangGraph's prebuilt ReAct pattern."""

    # Create Bedrock model
    model = ChatBedrockConverse(
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        temperature=0.1,
        region_name="us-east-1",
        max_tokens=8192,
    )

    # Generate API context for system prompt
    api_context = generate_api_context()

    # Create ReAct agent with minimal configuration
    return create_react_agent(
        model=model,
        tools=[search_chunks_tool, get_chunks_tool],
        prompt=f"""You are an expert API documentation researcher providing factually accurate information \
for developers.

{api_context}

CRITICAL REQUIREMENTS:
1. ALWAYS cite specific sources: Include chunk_id references for all claims
2. ONLY state what is explicitly documented - never infer or assume
3. Use exact field names, types, and values from the schemas
4. Include API endpoints exactly as documented
5. Quote error codes and messages precisely
6. When uncertain, state "not explicitly documented" rather than guess

Research Strategy:
1. Start with search_chunks_tool to find relevant chunks
2. Use file_filter when user mentions specific APIs ("sponsored-display", "dsp", etc.)
3. Use get_chunks_tool with expand_depth=3-5 to get complete schemas and context
4. Cross-reference multiple chunks to ensure accuracy

Response Format:
- Start each major point with [Source: chunk_id]
- Use exact API endpoint paths from documentation
- Include precise schema field names and types
- Quote exact error messages when relevant
- Provide working code examples only when all fields are documented

Tool Usage Guidelines:
- search_chunks_tool: Use file_filter for focused searches
- get_chunks_tool: expand_depth=3-5 for complete schemas, 5+ for deep nested structures

Remember: Developers need 100% accurate, verifiable information they can implement directly.""",
    )


async def research_api_question(question: str) -> str:
    """Research API documentation using ReAct agent."""
    logger.info(f"Starting research for question: {question[:100]}...")

    agent = create_research_agent()

    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # Count tool calls and iterations
    messages = result["messages"]
    tool_calls = 0
    iterations = 0

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls += len(msg.tool_calls)
            iterations += 1
            for tool_call in msg.tool_calls:
                logger.info(f"Tool called: {tool_call['name']} with args: {tool_call['args']}")

    logger.info(f"Research completed: {iterations} iterations, {tool_calls} tool calls, {len(messages)} total messages")

    return result["messages"][-1].content
