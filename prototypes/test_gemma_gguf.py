#!/usr/bin/env python3
"""
Test Gemma 3n E4B GGUF using llama-cpp-python
"""

from llama_cpp import Llama
import json
import os
from pathlib import Path

def download_gguf_model():
    """Download GGUF model if not present."""
    model_path = Path("models/gemma-3n-E4B-it-Q4_K_M.gguf")
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return str(model_path)
    
    # Create models directory
    model_path.parent.mkdir(exist_ok=True)
    
    print("🔄 Downloading Gemma 3n E4B GGUF model...")
    print("   This will download ~2GB model file...")
    
    # Use huggingface-hub to download
    try:
        from huggingface_hub import hf_hub_download
        
        downloaded_path = hf_hub_download(
            repo_id="unsloth/gemma-3n-E4B-it-GGUF",
            filename="gemma-3n-E4B-it-Q4_K_M.gguf",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def test_gemma_gguf():
    print("🔄 Testing Gemma 3n E4B GGUF with llama-cpp-python...")
    
    # Download model if needed
    model_path = download_gguf_model()
    if not model_path:
        return False
    
    try:
        # Initialize llama-cpp model
        print("🔄 Loading GGUF model with llama-cpp...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context size
            n_threads=4,  # Number of threads
            verbose=False
        )
        print("✅ GGUF model loaded successfully!")
        
        # Use the simple chat completion API
        messages = [
            {
                "role": "system",
                "content": "You are an API documentation expert. Expand user queries to improve search in OpenAPI documentation. Respond ONLY with valid JSON."
            },
            {
                "role": "user", 
                "content": """Expand this API documentation query: "How do I create a campaign?"

Output format: {"expanded_query": "...", "query_type": "endpoint", "key_terms": [...], "related_concepts": [...]}

Your response:"""
            }
        ]
        
        print("🔄 Testing query expansion...")
        
        # Generate response using chat completion API
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.3,
            top_p=0.95,
            stream=False  # Ensure we get a complete response, not a stream
        )
        
        response = output['choices'][0]['message']['content']
        if response:
            response = response.strip()
        else:
            response = ""
        print(f"🤖 Gemma response: {response}")
        
        # Try to parse JSON
        try:
            # Remove markdown code fences if present
            clean_response = response.replace('```json', '').replace('```', '').strip()
            
            # Look for JSON in cleaned response
            start_idx = clean_response.find('{')
            end_idx = clean_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = clean_response[start_idx:end_idx]
                
                # Handle incomplete JSON (common with truncated responses)
                if not json_text.endswith('}'):
                    # Count braces and brackets to try to complete JSON
                    open_braces = json_text.count('{') - json_text.count('}')
                    open_brackets = json_text.count('[') - json_text.count(']')
                    
                    # Add missing closing characters
                    json_text += ']' * open_brackets
                    json_text += '}' * open_braces
                
                # Try to fix common JSON issues
                if ',"' in json_text and not json_text.endswith('}'):
                    # Find last complete key-value pair
                    last_comma = json_text.rfind(',')
                    if last_comma > 0:
                        json_text = json_text[:last_comma] + '}'
                
                result = json.loads(json_text)
                print("✅ JSON parsing successful!")
                print(f"   Expanded query: {result.get('expanded_query', 'N/A')[:100]}...")
                print(f"   Query type: {result.get('query_type', 'N/A')}")
                print(f"   Key terms: {result.get('key_terms', [])[:5]}")
                return True
            else:
                print("⚠️ No JSON found in response")
                print("   Raw response for analysis:")
                print(f"   {response[:200]}...")
                return False
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print("   Raw response for analysis:")
            print(f"   {response}")
            return False
            
    except Exception as e:
        print(f"❌ Gemma GGUF test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gemma_gguf()
    if success:
        print("🎯 Gemma GGUF integration ready!")
    else:
        print("❌ Gemma GGUF integration needs adjustment")
