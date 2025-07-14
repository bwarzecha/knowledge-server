"""Tests for Research Agent retry configuration."""

from unittest.mock import Mock, patch

import pytest

from src.cli.config import Config
from src.research_agent.agent import create_research_agent
from src.research_agent.tools import _filter_and_expand_chunks


class TestRetryConfiguration:
    """Test retry configuration is properly applied."""

    def test_config_has_retry_settings(self):
        """Test that Config class has retry configuration."""
        config = Config()

        # Check that retry settings exist and have sensible defaults
        assert hasattr(config, "research_agent_llm_retry_max_attempts")
        assert hasattr(config, "chunk_filtering_llm_retry_max_attempts")
        assert config.research_agent_llm_retry_max_attempts >= 1
        assert config.chunk_filtering_llm_retry_max_attempts >= 1

    @patch("src.research_agent.agent.ChatBedrockConverse")
    def test_research_agent_uses_retry_config(self, mock_bedrock):
        """Test that research agent LLM is configured with retry settings."""
        # Setup
        config = Config()

        # Execute
        create_research_agent()

        # Verify ChatBedrockConverse was called with config parameter
        mock_bedrock.assert_called_once()
        call_args = mock_bedrock.call_args
        assert "config" in call_args.kwargs

        # Verify the config has retry settings
        aws_config = call_args.kwargs["config"]
        assert aws_config.retries["max_attempts"] == config.research_agent_llm_retry_max_attempts
        assert aws_config.retries["mode"] == "standard"

    @patch("src.research_agent.tools.ChatBedrockConverse")
    @pytest.mark.asyncio
    async def test_chunk_filtering_uses_retry_config(self, mock_bedrock):
        """Test that chunk filtering LLM is configured with retry settings."""
        # Setup
        config = Config()
        mock_vector_store = Mock()
        mock_chunks = []

        # Mock the LLM to avoid actual calls
        mock_model = Mock()
        mock_model.ainvoke.return_value = Mock(content="test response")
        mock_bedrock.return_value = mock_model

        # Execute
        try:
            await _filter_and_expand_chunks(
                chunks=mock_chunks,
                query="test query",
                search_context=None,
                target_count=5,
                vector_store=mock_vector_store,
                config=config,
            )
        except Exception:
            # We expect this to fail due to mocking, but we're testing the config setup
            pass

        # Verify ChatBedrockConverse was called with config parameter
        mock_bedrock.assert_called_once()
        call_args = mock_bedrock.call_args
        assert "config" in call_args.kwargs

        # Verify the config has retry settings
        aws_config = call_args.kwargs["config"]
        assert aws_config.retries["max_attempts"] == config.chunk_filtering_llm_retry_max_attempts
        assert aws_config.retries["mode"] == "standard"

    def test_retry_config_from_environment(self):
        """Test that retry settings can be configured via environment variables."""
        import os

        # Test with custom environment values
        test_env = {"RESEARCH_AGENT_LLM_RETRY_MAX_ATTEMPTS": "5", "CHUNK_FILTERING_LLM_RETRY_MAX_ATTEMPTS": "2"}

        with patch.dict(os.environ, test_env):
            config = Config()
            assert config.research_agent_llm_retry_max_attempts == 5
            assert config.chunk_filtering_llm_retry_max_attempts == 2
