"""
Configuration management with Pydantic settings + YAML.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetryConfig(BaseModel):
    """Retry policy configuration."""
    max_attempts: int = 5
    initial_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_500: bool = True
    retry_on_overloaded: bool = True
    retry_on_rate_limit: bool = True


class TimeoutConfig(BaseModel):
    """Timeout configuration."""
    turn_timeout: float = 300.0  # 5 minutes per turn
    tool_timeout: float = 120.0  # 2 minutes per tool
    session_timeout: float = 3600.0  # 1 hour total session


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    permission_mode: str = "acceptEdits"


class Settings(BaseSettings):
    """Main settings loaded from environment."""
    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Langfuse
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="glow", alias="MONGODB_DATABASE")

    # Loaded from YAML
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_yaml_config(path: Path | str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    """Get settings with YAML overrides."""
    yaml_config = load_yaml_config()

    settings = Settings()

    if "retry" in yaml_config:
        settings.retry = RetryConfig(**yaml_config["retry"])
    if "timeout" in yaml_config:
        settings.timeout = TimeoutConfig(**yaml_config["timeout"])
    if "agent" in yaml_config:
        settings.agent = AgentConfig(**yaml_config["agent"])

    return settings


# Global settings instance
settings = get_settings()
