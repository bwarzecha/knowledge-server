"""Research Agent using LangGraph's create_react_agent."""

import logging

from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

from ..cli.config import Config
from .agent_tools import get_chunks_tool, search_chunks_tool
from .tools import generate_api_context

logger = logging.getLogger(__name__)


def create_research_agent():
    """Create Research Agent using LangGraph's prebuilt ReAct pattern."""

    # Load configuration
    config = Config()

    # Configure retry behavior at AWS SDK level
    aws_config = BotocoreConfig(
        retries={
            "max_attempts": config.research_agent_llm_retry_max_attempts,
            "mode": "standard",
        }
    )

    # Create Bedrock model with configurable model and token limit
    model = ChatBedrockConverse(
        model=config.research_agent_llm_model,
        temperature=0.1,
        max_tokens=config.research_agent_llm_max_tokens,
        config=aws_config,
    )

    # Generate API context for system prompt
    api_context = generate_api_context()

    # Create ReAct agent with minimal configuration
    return create_react_agent(
        model=model,
        tools=[search_chunks_tool, get_chunks_tool],
        prompt=f"""You are an expert API documentation researcher providing comprehensive, detailed information \
for developers who need complete, implementable specifications.

{api_context}

RESPONSE REQUIREMENTS:
1. Provide COMPLETE information including:
   - FULL schema definitions with ALL fields, not summaries
   - Data types, constraints, and validation rules for each field
   - All enum values with their meanings
   - Examples from the documentation
   - Hierarchical relationships between components
   - All configuration options with detailed explanations

2. For schemas and complex structures:
   - Include the ENTIRE schema in code blocks, not summaries
   - Show nested structures with proper indentation
   - Include all properties, even optional ones
   - Provide field descriptions and requirements
   - Show inheritance relationships and references

3. Source citations:
   - Include [Source: chunk_id] for all information
   - Reference multiple sources when needed for completeness

4. Research approach:
   - Use search_chunks_tool to find ALL relevant content
   - Use get_chunks_tool with expand_depth=5-10 for complete schemas
   - Include ALL relevant information from expanded chunks
   - Never summarize or truncate schema content

5. Formatting requirements:
   ```json
   // Use code blocks for ALL schemas
   {{
     "property": "Show complete structure",
     "nested": {{
       "subProperty": "Include all levels"
     }}
   }}
   ```
   - Use bullet points for field explanations
   - Maintain hierarchical structure in explanations
   - Include examples in code blocks

6. Content priorities:
   - Completeness over brevity - developers need ALL details
   - Include edge cases and special configurations
   - Show all allowed values and their implications
   - Explain relationships between different API elements

Example of expected detail level:
Instead of: "The campaign has a targeting type field"
Provide:
```json
{{
  "targetingType": {{
    "type": "string",
    "enum": ["MANUAL", "AUTO"],
    "description": "Determines how targets are selected",
    "required": true,
    "details": {{
      "MANUAL": "Advertiser specifies keywords/products",
      "AUTO": "Amazon automatically targets based on product"
    }}
  }}
}}
```

Tool Usage:
- search_chunks_tool: Cast a wide net to find ALL related information
- get_chunks_tool: Use expand_depth=10 to ensure complete context
- Process multiple chunks to build comprehensive response

Remember: Developers are implementing based on your response. They need EVERY field, EVERY constraint, \
EVERY configuration option. When in doubt, include MORE detail, not less.""",
    )


async def research_api_question(question: str, exclude_chunks: str = "") -> str:
    """Research API documentation using ReAct agent."""
    logger.info(f"Starting research for question: {question[:100]}...")

    # If exclude_chunks provided, add context to the agent about excluding chunks
    full_question = question
    if exclude_chunks.strip():
        full_question = f"{question}\n\nIMPORTANT: Exclude these chunk IDs from all search results: {exclude_chunks}"

    agent = create_research_agent()

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": full_question}]}
    )

    # Count tool calls and iterations
    messages = result["messages"]
    tool_calls = 0
    iterations = 0

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls += len(msg.tool_calls)
            iterations += 1
            for tool_call in msg.tool_calls:
                logger.info(
                    f"Tool called: {tool_call['name']} with args: {tool_call['args']}"
                )

    logger.info(
        f"Research completed: {iterations} iterations, {tool_calls} tool calls, {len(messages)} total messages"
    )

    return result["messages"][-1].content
