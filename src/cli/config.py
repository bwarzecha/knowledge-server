"""Configuration management for knowledge server CLI."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Configuration loaded from .env file."""

    def __init__(self, env_file: Optional[str] = None):
        """Load configuration from .env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Load from project root .env
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)

        # OpenAPI Processing
        self.openapi_specs_dir = os.getenv("OPENAPI_SPECS_DIR", "./samples")
        self.skip_hidden_files = os.getenv("SKIP_HIDDEN_FILES", "true").lower() == "true"
        self.supported_extensions = os.getenv("SUPPORTED_EXTENSIONS", ".json,.yaml,.yml").split(",")
        self.log_processing_progress = os.getenv("LOG_PROCESSING_PROGRESS", "true").lower() == "true"

        # Markdown Processing
        self.markdown_source_dir = os.getenv("MARKDOWN_SOURCE_DIR", self.openapi_specs_dir)
        self.markdown_max_tokens = int(os.getenv("MARKDOWN_MAX_TOKENS", "1000"))
        self.markdown_file_extensions = os.getenv("MARKDOWN_FILE_EXTENSIONS", ".md,.markdown").split(",")

        # Vector Store
        self.vector_store_dir = os.getenv("VECTOR_STORE_DIR", "./data/vectorstore")
        self.vector_store_collection = os.getenv("VECTOR_STORE_COLLECTION", "knowledge_base")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "dunzhang/stella_en_1.5B_v5")
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "mps")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8192"))

        # API Index
        self.api_index_path = os.getenv("API_INDEX_PATH", "./data/api_index.json")

        # MCP Server
        self.mcp_server_name = os.getenv("MCP_SERVER_NAME", "Knowledge Server")
        self.mcp_server_host = os.getenv("MCP_SERVER_HOST", "localhost")
        self.mcp_server_port = int(os.getenv("MCP_SERVER_PORT", "8000"))

        # Knowledge Retriever Configuration
        self.retrieval_max_primary_results = int(os.getenv("RETRIEVAL_MAX_PRIMARY_RESULTS", "5"))
        self.retrieval_max_total_chunks = int(os.getenv("RETRIEVAL_MAX_TOTAL_CHUNKS", "15"))
        self.retrieval_max_depth = int(os.getenv("RETRIEVAL_MAX_DEPTH", "3"))
        self.retrieval_timeout_ms = int(os.getenv("RETRIEVAL_TIMEOUT_MS", "5000"))
        self.context_token_limit = int(os.getenv("CONTEXT_TOKEN_LIMIT", "4000"))
        self.context_prioritize_primary = os.getenv("CONTEXT_PRIORITIZE_PRIMARY", "true").lower() == "true"

        # Intelligent Chunk Filtering Configuration
        self.chunk_filtering_oversample_multiplier = float(os.getenv("CHUNK_FILTERING_OVERSAMPLE_MULTIPLIER", "1.6"))
        self.chunk_filtering_llm_timeout_ms = int(os.getenv("CHUNK_FILTERING_LLM_TIMEOUT_MS", "10000"))

        # Re-ranker LLM Configuration (separate from main research agent)
        self.reranker_llm_provider = os.getenv("RERANKER_LLM_PROVIDER", "bedrock")
        self.reranker_llm_model = os.getenv("RERANKER_LLM_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
        self.reranker_llm_region = os.getenv("RERANKER_LLM_REGION", "us-east-1")
        self.reranker_llm_temperature = float(os.getenv("RERANKER_LLM_TEMPERATURE", "0.1"))
        self.reranker_llm_max_tokens = int(
            os.getenv("RERANKER_LLM_MAX_TOKENS", "2048")
        )  # Sufficient for decision list output

        # Research Agent LLM Configuration (main agent, not re-ranker)
        self.research_agent_llm_model = os.getenv(
            "RESEARCH_AGENT_LLM_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        self.research_agent_llm_max_tokens = int(os.getenv("RESEARCH_AGENT_LLM_MAX_TOKENS", "40000"))
        self.research_agent_llm_retry_max_attempts = int(os.getenv("RESEARCH_AGENT_LLM_RETRY_MAX_ATTEMPTS", "3"))

        # Chunk Filtering LLM Retry Configuration
        self.chunk_filtering_llm_retry_max_attempts = int(os.getenv("CHUNK_FILTERING_LLM_RETRY_MAX_ATTEMPTS", "3"))

        # Validation
        self.min_openapi_version = os.getenv("MIN_OPENAPI_VERSION", "3.0.0")
        self.require_info_section = os.getenv("REQUIRE_INFO_SECTION", "true").lower() == "true"
        self.require_paths_or_components = os.getenv("REQUIRE_PATHS_OR_COMPONENTS", "true").lower() == "true"

    def ensure_data_dirs(self):
        """Ensure data directories exist."""
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
        Path(self.api_index_path).parent.mkdir(parents=True, exist_ok=True)
