#!/usr/bin/env python3
"""
Test Gemma 3n E4B GGUF integration
"""

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import json

def test_gemma():
    print("🔄 Testing Gemma 3n E4B GGUF...")
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA")
    elif torch.backends.mps.is_available():
        device = "mps" 
        print("🍎 Using MPS")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    try:
        # Initialize model from GGUF
        print("🔄 Loading Gemma 3n E4B GGUF model...")
        model_id = "unsloth/gemma-3n-E4B-it-GGUF"
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_id)
        print("✅ Model loaded successfully!")
        
        # Test query expansion
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an API documentation expert. Expand user queries to improve search in OpenAPI documentation. Respond ONLY with valid JSON."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """Expand this API documentation query: "How do I create a campaign?"

Output format: {"expanded_query": "...", "query_type": "endpoint", "key_terms": [...], "related_concepts": [...]}

Your response:"""}
                ]
            }
        ]
        
        print("🔄 Testing query expansion...")
        
        # Apply chat template and generate
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )
            generation = generation[0][input_len:]
            response = processor.decode(generation, skip_special_tokens=True)
        
        print(f"🤖 Gemma response: {response}")
        
        # Try to parse JSON
        try:
            # Look for JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = response[start_idx:end_idx]
                result = json.loads(json_text)
                print("✅ JSON parsing successful!")
                print(f"   Expanded query: {result.get('expanded_query', 'N/A')}")
                print(f"   Query type: {result.get('query_type', 'N/A')}")
                return True
            else:
                print("⚠️ No JSON found in response")
                print("   Raw response for analysis:")
                print(f"   {response}")
                return False
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print("   Raw response for analysis:")
            print(f"   {response}")
            return False
            
    except Exception as e:
        print(f"❌ Gemma test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gemma()
    if success:
        print("🎯 Gemma integration ready!")
    else:
        print("❌ Gemma integration needs adjustment")