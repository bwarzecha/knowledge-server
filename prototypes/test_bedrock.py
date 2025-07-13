#!/usr/bin/env python3
"""
Test AWS Bedrock Claude Haiku integration
"""

import json
import boto3

def test_bedrock_claude():
    client = boto3.client('bedrock-runtime')
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Test query expansion
    system_prompt = """You are an API documentation expert. Your task is to expand user queries to improve search in OpenAPI documentation.

Given a user query, provide:
1. expanded_query: The original query plus related API terms, synonyms, and concepts
2. query_type: One of 'endpoint', 'schema', 'error', 'concept'
3. key_terms: Important terms that should be weighted higher in search
4. related_concepts: Related API concepts that might be relevant

Focus on API/OpenAPI terminology like endpoints, operations, schemas, responses, parameters, etc.

Respond with valid JSON only."""

    user_prompt = """Expand this API documentation query: "How do I create a campaign?"

Example:
Query: "How do I create a campaign?"
Response: {
  "expanded_query": "How do I create campaign POST endpoint createCampaign operation request body parameters required fields",
  "query_type": "endpoint", 
  "key_terms": ["create", "campaign", "POST", "createCampaign"],
  "related_concepts": ["campaign management", "campaign creation", "POST request", "create operation"]
}

Now expand: "How do I create a campaign?" """

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
        
        print("🔄 Testing AWS Bedrock Claude Haiku...")
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text'].strip()
        
        print("✅ Response received:")
        print(content)
        
        # Try to parse as JSON
        result = json.loads(content)
        print("✅ JSON parsing successful!")
        print(f"Expanded query: {result.get('expanded_query', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bedrock test failed: {e}")
        return False

if __name__ == "__main__":
    test_bedrock_claude()