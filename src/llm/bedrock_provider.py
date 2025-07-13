"""AWS Bedrock LLM provider using boto3."""

import json
import logging

from .provider import LLMProvider

logger = logging.getLogger(__name__)


class BedrockLLMProvider(LLMProvider):
    """AWS Bedrock LLM provider using boto3."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    ):
        self.region = region
        self.model_id = model_id
        self._client = None
        self._available = None

    def _initialize_client(self):
        """Initialize Bedrock client."""
        if self._client is not None:
            return

        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            self._client = boto3.client("bedrock-runtime", region_name=self.region)

            # Test connectivity with a simple call
            try:
                bedrock_client = boto3.client("bedrock", region_name=self.region)
                bedrock_client.list_foundation_models()
                self._available = True
            except ClientError:
                # Fallback: assume available if client creation succeeds
                self._available = True

        except ImportError:
            logger.warning("Warning: boto3 not installed")
            self._available = False
        except NoCredentialsError:
            logger.warning("Warning: AWS credentials not configured")
            self._available = False
        except Exception as e:
            logger.warning(f"Warning: Failed to initialize Bedrock: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if Bedrock is available."""
        if self._available is None:
            self._initialize_client()
        return self._available or False

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> str:
        """Generate text using Bedrock."""
        if not self.is_available():
            raise RuntimeError("Bedrock is not available")

        if self._client is None:
            self._initialize_client()

        try:
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31",
            }

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"].strip()

        except Exception as e:
            raise RuntimeError(f"Bedrock generation failed: {e}")
