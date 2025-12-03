"""Langfuse integration for LLM observability."""

from typing import Any
from contextlib import contextmanager

from config import settings


class LangfuseTracer:
    """
    Langfuse tracer for LLM observability.

    Disabled by default until LANGFUSE_ENABLED=true and credentials configured.
    """

    def __init__(self):
        self._client = None
        self._enabled = settings.langfuse_enabled

        if self._enabled:
            self._init_client()

    def _init_client(self):
        """Initialize Langfuse client if enabled."""
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host or None,
            )
        except ImportError:
            self._enabled = False
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def trace(
        self,
        session_id: str,
        user_id: str | None = None,
        name: str = "agent_run",
        metadata: dict[str, Any] | None = None,
    ):
        """Create a trace for a session."""
        if not self.enabled:
            return NoOpTrace()

        return self._client.trace(
            session_id=session_id,
            user_id=user_id,
            name=name,
            metadata=metadata or {},
        )

    def generation(
        self,
        trace,
        name: str,
        model: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
    ):
        """Create a generation span within a trace."""
        if not self.enabled or isinstance(trace, NoOpTrace):
            return NoOpGeneration()

        return trace.generation(
            name=name,
            model=model,
            input=input,
            metadata=metadata or {},
        )

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled:
            self._client.flush()


class NoOpTrace:
    """No-op trace when Langfuse is disabled."""

    def generation(self, **kwargs):
        return NoOpGeneration()

    def span(self, **kwargs):
        return NoOpSpan()

    def event(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class NoOpGeneration:
    """No-op generation when Langfuse is disabled."""

    def end(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class NoOpSpan:
    """No-op span when Langfuse is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def end(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


# Global tracer instance
tracer = LangfuseTracer()
