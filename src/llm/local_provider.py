"""Local LLM provider using llama-cpp-python."""

import logging

from .provider import LLMProvider

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python."""

    def __init__(
        self,
        repo_id: str = "unsloth/gemma-3-4b-it-GGUF",
        filename: str = "*Q4_K_S.gguf",
        n_ctx: int = 8192,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self._llm = None
        self._available = None

    def _initialize_llm(self):
        """Initialize llama-cpp-python model."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading model from {self.repo_id}/{self.filename}...")
            self._llm = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            self._available = True
            logger.info("✅ Model loaded successfully")

        except ImportError:
            logger.info("Warning: llama-cpp-python not installed")
            self._available = False
        except Exception as e:
            logger.info(f"Warning: Failed to load model: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if local LLM is available."""
        if self._available is None:
            self._initialize_llm()
        return self._available or False

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> str:
        """Generate text using local LLM."""
        if not self.is_available():
            raise RuntimeError("Local LLM is not available")

        if self._llm is None:
            self._initialize_llm()

        try:
            # Use chat completion API like the working prototype
            messages = [{"role": "user", "content": prompt}]

            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

            content = response["choices"][0]["message"]["content"]
            return content.strip() if content else ""

        except Exception as e:
            raise RuntimeError(f"Local LLM generation failed: {e}")
