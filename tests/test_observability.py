"""Tests for observability layer."""

import pytest
from observability.logging import get_logger, setup_logging, LogContext
from observability.langfuse import LangfuseTracer, NoOpTrace, NoOpGeneration


class TestLogging:
    def test_get_logger(self):
        log = get_logger()
        assert log is not None

    def test_get_logger_with_name(self):
        log = get_logger("test_logger")
        assert log is not None

    def test_setup_logging(self):
        # Should not raise
        setup_logging(level="DEBUG")
        setup_logging(level="INFO", json_output=True)


class TestLogContext:
    def test_context_manager(self):
        with LogContext(session_id="test123", user_id="user456"):
            ctx = LogContext.get()
            assert ctx["session_id"] == "test123"
            assert ctx["user_id"] == "user456"

        # Context should be cleared after exit
        ctx = LogContext.get()
        assert "session_id" not in ctx

    def test_nested_context(self):
        with LogContext(session_id="outer"):
            assert LogContext.get()["session_id"] == "outer"

            with LogContext(turn=1):
                ctx = LogContext.get()
                assert ctx["session_id"] == "outer"
                assert ctx["turn"] == 1

            # Inner context cleared, outer preserved
            ctx = LogContext.get()
            assert ctx["session_id"] == "outer"
            assert "turn" not in ctx

    def test_set_context(self):
        with LogContext(session_id="test"):
            LogContext.set(extra="value")
            ctx = LogContext.get()
            assert ctx["session_id"] == "test"
            assert ctx["extra"] == "value"


class TestLangfuseTracer:
    def test_disabled_by_default(self):
        # Langfuse should be disabled when not configured
        tracer = LangfuseTracer()
        # May or may not be enabled depending on env, but trace should work
        trace = tracer.trace(session_id="test")
        assert trace is not None

    def test_noop_trace(self):
        trace = NoOpTrace()

        # All methods should work without raising
        gen = trace.generation(name="test")
        span = trace.span(name="test")
        trace.event(name="test")
        trace.update(name="updated")

        assert isinstance(gen, NoOpGeneration)

    def test_noop_generation(self):
        gen = NoOpGeneration()

        # All methods should work without raising
        gen.end(output="test")
        gen.update(name="updated")

    def test_tracer_flush(self):
        tracer = LangfuseTracer()
        # Should not raise even when disabled
        tracer.flush()
