"""
REAL Proof of Concept: LLaMA.cpp Tool Calling

This tests if ChatLlamaCpp can:
1. Bind tools using .bind_tools()
2. Automatically call tools when the LLM decides to
3. Work with create_react_agent
4. Handle real tool responses

Testing with Gemma model as requested.
"""

import asyncio
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

def create_gemma_model():
    """Helper function to create Gemma model - avoid code duplication"""
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        from llama_cpp import Llama
        
        print("📥 Loading Gemma model from HuggingFace...")
        
        # Load base model from HuggingFace
        base_llm = Llama.from_pretrained(
            repo_id="unsloth/gemma-3-4b-it-GGUF",
            filename="*Q4_K_S.gguf", 
            n_ctx=2048,
            verbose=False
        )
        
        print("✅ Base model loaded, creating ChatLlamaCpp wrapper...")
        
        # ChatLlamaCpp expects model_path, not llm parameter
        # Let's try a different approach
        model = ChatLlamaCpp(
            model_path=None,  # Required field but we'll override
            temperature=0.1,
            max_tokens=500,
            verbose=False
        )
        
        # Override the internal llm
        model.client = base_llm
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_llamacpp_tool_binding():
    """Test if ChatLlamaCpp can bind tools and call them"""
    print("\n🧪 Test 1: Direct Tool Binding with ChatLlamaCpp")
    print("-" * 60)
    
    try:
        from langchain_community.llms import LlamaCpp
        from langchain_community.chat_models import ChatLlamaCpp
        
        # Create LLaMA.cpp model (using Gemma from HuggingFace)
        # Try HuggingFace repo approach like in local_provider.py
        try:
            from llama_cpp import Llama
            
            # Load model from HuggingFace like the existing code
            base_llm = Llama.from_pretrained(
                repo_id="unsloth/gemma-3-4b-it-GGUF",
                filename="*Q4_K_S.gguf", 
                n_ctx=2048,
                verbose=False
            )
            
            # Create ChatLlamaCpp wrapper
            model = ChatLlamaCpp(
                llm=base_llm,
                temperature=0.1,
                max_tokens=500,
                verbose=False
            )
        except Exception as e:
            print(f"❌ HuggingFace approach failed: {e}")
            # Fallback to direct model path approach
            model = ChatLlamaCpp(
                model_path="local/models/gemma-2-2b-it-q4_k_m.gguf",
                temperature=0.1,
                max_tokens=500,
                n_ctx=2048,
                verbose=False
            )
        
        print("✅ ChatLlamaCpp model created")
        
        # Try to bind tools
        tools = [simple_math_tool, weather_tool]
        
        # Check if bind_tools method exists
        if hasattr(model, 'bind_tools'):
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
        else:
            print("❌ ChatLlamaCpp does not support bind_tools method")
            print("🔧 Will need to use manual tool calling approach")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("💡 Please check if the Gemma model file exists at the specified path")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_llamacpp_react_agent():
    """Test if create_react_agent works with ChatLlamaCpp"""
    print("\n🧪 Test 2: create_react_agent with ChatLlamaCpp")
    print("-" * 60)
    
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        
        # Create LLaMA.cpp model (using Gemma from HuggingFace)
        try:
            from llama_cpp import Llama
            
            # Load model from HuggingFace like the existing code
            base_llm = Llama.from_pretrained(
                repo_id="unsloth/gemma-3-4b-it-GGUF",
                filename="*Q4_K_S.gguf", 
                n_ctx=2048,
                verbose=False
            )
            
            # Create ChatLlamaCpp wrapper
            model = ChatLlamaCpp(
                llm=base_llm,
                temperature=0.1,
                max_tokens=500,
                verbose=False
            )
        except Exception as e:
            print(f"❌ HuggingFace approach failed: {e}")
            return False
        
        # Try to create ReAct agent
        try:
            agent = create_react_agent(
                model=model,
                tools=[simple_math_tool, weather_tool],
                prompt="You are a helpful assistant. Use tools when appropriate."
            )
            
            print("✅ ReAct agent created")
            
            # Test agent with math question
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "What is 23 + 45?"}]
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
            
            final_answer = result["messages"][-1].content
            print(f"\n📝 Final Answer: {final_answer}")
            
            return True
            
        except Exception as e:
            print(f"❌ create_react_agent failed: {e}")
            print("🔧 ChatLlamaCpp might not be compatible with ReAct agent")
            return False
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("💡 Please check if the Gemma model file exists")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_llamacpp_manual_tools():
    """Test manual tool calling approach for LLaMA.cpp"""
    print("\n🧪 Test 3: Manual Tool Calling with ChatLlamaCpp")
    print("-" * 60)
    
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        
        # Create LLaMA.cpp model (using Gemma from HuggingFace)
        try:
            from llama_cpp import Llama
            
            # Load model from HuggingFace like the existing code
            base_llm = Llama.from_pretrained(
                repo_id="unsloth/gemma-3-4b-it-GGUF",
                filename="*Q4_K_S.gguf", 
                n_ctx=2048,
                verbose=False
            )
            
            # Create ChatLlamaCpp wrapper
            model = ChatLlamaCpp(
                llm=base_llm,
                temperature=0.1,
                max_tokens=500,
                verbose=False
            )
        except Exception as e:
            print(f"❌ HuggingFace approach failed: {e}")
            return False
        
        print("✅ ChatLlamaCpp model created")
        
        # Manual prompt with tool descriptions
        prompt = """You are a helpful assistant with access to tools. When you need to use a tool, respond with exactly this format:

TOOL_CALL: tool_name(arg1=value1, arg2=value2)

Available tools:
- simple_math_tool(a: int, b: int) -> int: Add two numbers together
- weather_tool(city: str) -> str: Get weather for a city

User: What is 15 + 27?
Assistant:"""

        response = model.invoke(prompt)
        print("✅ Model response received")
        print(f"📝 Response: {response.content}")
        
        # Check if response contains tool call pattern
        if "TOOL_CALL:" in response.content and "simple_math_tool" in response.content:
            print("✅ Model correctly identified need for tool call")
            print("🔧 Manual tool calling approach would work")
            return True
        else:
            print("❌ Model did not use expected tool call format")
            print("🔧 May need different prompting strategy")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_environment():
    """Check if environment is set up correctly"""
    print("🔧 Environment Check:")
    print(f"  - Python version: {sys.version.split()[0]}")
        
    # Check package imports
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        from langgraph.prebuilt import create_react_agent
        print("  - Required packages: ✅ Available")
        return True
    except ImportError as e:
        print(f"  - Required packages: ❌ Missing ({e})")
        return False

async def main():
    """Run the LLaMA.cpp tool calling tests"""
    print("🚀 LLaMA.cpp Tool Calling Proof of Concept")
    print("=" * 60)
    
    # Environment check
    if not test_environment():
        print("\n❌ Environment not ready. Please fix the issues above.")
        return False
    
    results = []
    
    # Test 1: Direct tool binding (might fail)
    results.append(test_llamacpp_tool_binding())
    
    # Test 2: ReAct agent (might fail if tool binding fails)
    results.append(await test_llamacpp_react_agent())
    
    # Test 3: Manual tool calling (fallback approach)
    results.append(test_llamacpp_manual_tools())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("-" * 30)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if results[0] and results[1]:  # Native tool calling works
        print("🎉 SUCCESS! LLaMA.cpp supports native tool calling!")
        print("💡 Can use same approach as AWS Bedrock")
    elif results[2]:  # Manual approach works
        print("🔧 PARTIAL SUCCESS! LLaMA.cpp needs manual tool calling")
        print("💡 Will need custom tool orchestration for production")
    else:
        print("❌ FAILED! LLaMA.cpp tool calling has issues.")
        print("🔧 Need to investigate alternative approaches")
    
    return passed > 0

if __name__ == "__main__":
    asyncio.run(main())