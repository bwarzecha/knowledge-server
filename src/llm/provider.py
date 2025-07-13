"""LLM provider abstraction and factory."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> str:
        """Generate text from prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""


class LLMClient:
    """Main interface for LLM operations."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> str:
        """Generate text using configured provider."""
        if not self.provider.is_available():
            raise RuntimeError(f"LLM provider {type(self.provider).__name__} is not available")

        return self.provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.provider.is_available()


def create_llm_client(provider_type: str = "local", **config) -> LLMClient:
    """Factory function to create LLM client with specified provider."""

    if provider_type == "local":
        from .local_provider import LocalLLMProvider

        return LLMClient(LocalLLMProvider(**config))

    elif provider_type == "bedrock":
        from .bedrock_provider import BedrockLLMProvider

        return LLMClient(BedrockLLMProvider(**config))

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
