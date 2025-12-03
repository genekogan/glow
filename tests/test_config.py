"""Tests for configuration."""

import pytest
import tempfile
import os
from pathlib import Path

from config import (
    Settings, RetryConfig, TimeoutConfig, AgentConfig,
    load_yaml_config, get_settings,
)


class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.retry_on_500 is True

    def test_custom_values(self):
        config = RetryConfig(max_attempts=10, initial_delay=1.0)
        assert config.max_attempts == 10
        assert config.initial_delay == 1.0


class TestTimeoutConfig:
    def test_defaults(self):
        config = TimeoutConfig()
        assert config.turn_timeout == 300.0
        assert config.tool_timeout == 120.0
        assert config.session_timeout == 3600.0

    def test_custom_values(self):
        config = TimeoutConfig(turn_timeout=600.0)
        assert config.turn_timeout == 600.0


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_turns == 50
        assert config.permission_mode == "acceptEdits"


class TestLoadYamlConfig:
    def test_missing_file(self):
        result = load_yaml_config("nonexistent.yaml")
        assert result == {}

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            result = load_yaml_config(f.name)
        os.unlink(f.name)
        assert result == {}

    def test_valid_yaml(self):
        yaml_content = """
retry:
  max_attempts: 10
  initial_delay: 1.0
timeout:
  turn_timeout: 600.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            result = load_yaml_config(f.name)
        os.unlink(f.name)

        assert result["retry"]["max_attempts"] == 10
        assert result["retry"]["initial_delay"] == 1.0
        assert result["timeout"]["turn_timeout"] == 600.0


class TestSettings:
    def test_defaults(self):
        # Create settings without loading env
        settings = Settings(
            _env_file=None,
            ANTHROPIC_API_KEY="test-key",
        )
        assert settings.anthropic_api_key == "test-key"
        assert settings.langfuse_enabled is False
        assert settings.mongodb_uri == "mongodb://localhost:27017"

    def test_retry_config(self):
        settings = Settings(_env_file=None)
        assert isinstance(settings.retry, RetryConfig)
        assert settings.retry.max_attempts == 5

    def test_timeout_config(self):
        settings = Settings(_env_file=None)
        assert isinstance(settings.timeout, TimeoutConfig)
        assert settings.timeout.turn_timeout == 300.0

    def test_agent_config(self):
        settings = Settings(_env_file=None)
        assert isinstance(settings.agent, AgentConfig)
        assert settings.agent.model == "claude-sonnet-4-20250514"
