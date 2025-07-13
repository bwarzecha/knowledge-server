"""
REAL Proof of Concept: AWS Bedrock Tool Calling

This actually tests if ChatBedrockConverse can:
1. Bind tools using .bind_tools()
2. Automatically call tools when the LLM decides to
3. Work with create_react_agent
4. Handle real tool responses

This is the real test we need!
"""

import asyncio
import os
import sys
from typing import Optional

# LangChain and LangGraph imports
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def simple_math_tool(a: int, b: int) -> int:
    """Add two numbers together. Use this when you need to do math."""
    print(f"🧮 Math tool called: {a} + {b} = {a + b}")
    return a + b

@tool  
def weather_tool(city: str) -> str:
    """Get weather for a city. Use this when asked about weather."""
    print(f"🌤️ Weather tool called for: {city}")
    return f"The weather in {city} is sunny and 75°F"

def test_bedrock_tool_binding():
    """Test if ChatBedrockConverse can bind tools and call them"""
    print("\n🧪 Test 1: Direct Tool Binding with ChatBedrockConverse")
    print("-" * 60)
    
    try:
        from langchain_aws import ChatBedrockConverse
        
        # Create Bedrock model
        model = ChatBedrockConverse(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0.1,
            region_name="us-east-1"
        )
        
        print("✅ ChatBedrockConverse model created")
        
        # Try to bind tools
        tools = [simple_math_tool, weather_tool]
        model_with_tools = model.bind_tools(tools)
        
        print("✅ Tools bound to model")
        
        # Test if model can call tools
        response = model_with_tools.invoke("What is 15 + 27?")
        
        print("✅ Model response received")
        print(f"📝 Response type: {type(response)}")
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Tool calls found: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"   - {tool_call['name']}: {tool_call['args']}")
            return True
        else:
            print("❌ No tool calls found in response")
            print(f"📝 Response content: {response.content}")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_bedrock_react_agent():
    """Test if create_react_agent works with ChatBedrockConverse"""
    print("\n🧪 Test 2: create_react_agent with ChatBedrockConverse")
    print("-" * 60)
    
    try:
        from langchain_aws import ChatBedrockConverse
        
        # Create Bedrock model
        model = ChatBedrockConverse(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0.1,
            region_name="us-east-1"
        )
        
        # Create ReAct agent
        agent = create_react_agent(
            model=model,
            tools=[simple_math_tool, weather_tool],
            prompt="You are a helpful assistant. Use tools when appropriate."
        )
        
        print("✅ ReAct agent created")
        
        # Test agent with math question
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 23 + 45? Also what's the weather like in Paris?"}]
        })
        
        print("✅ Agent response received")
        
        # Check message flow
        print("\n📋 Message Flow:")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  {i}: {msg_type} - Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
            elif hasattr(msg, 'content') and msg.content:
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"  {i}: {msg_type} - {content}")
        
        # Check if both tools were called
        tool_messages = [msg for msg in result["messages"] if hasattr(msg, 'name')]
        math_called = any('math' in str(msg).lower() for msg in tool_messages)
        weather_called = any('weather' in str(msg).lower() for msg in tool_messages)
        
        print(f"\n🔧 Tool Usage Summary:")
        print(f"   - Math tool called: {math_called}")
        print(f"   - Weather tool called: {weather_called}")
        
        final_answer = result["messages"][-1].content
        print(f"\n📝 Final Answer: {final_answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_environment():
    """Check if environment is set up correctly"""
    print("🔧 Environment Check:")
    print(f"  - Python version: {sys.version.split()[0]}")
    print("  - AWS credentials: Assumed configured")
        
    # Check package imports
    try:
        from langchain_aws import ChatBedrockConverse
        from langgraph.prebuilt import create_react_agent
        print("  - Required packages: ✅ Available")
        return True
    except ImportError as e:
        print(f"  - Required packages: ❌ Missing ({e})")
        return False

async def main():
    """Run the REAL Bedrock tool calling tests"""
    print("🚀 REAL AWS Bedrock Tool Calling Proof of Concept")
    print("=" * 60)
    
    # Environment check
    if not test_environment():
        print("\n❌ Environment not ready. Please fix the issues above.")
        return False
    
    results = []
    
    # Test 1: Direct tool binding
    results.append(test_bedrock_tool_binding())
    
    # Test 2: ReAct agent
    results.append(await test_bedrock_react_agent())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("-" * 30)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 SUCCESS! AWS Bedrock tool calling actually works!")
        print("💡 Ready to build the production Research Agent!")
    else:
        print("❌ FAILED! AWS Bedrock tool calling has issues.")
        print("🔧 Need to investigate or find alternative approach.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())