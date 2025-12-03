"""Structured logging with session-scoped context."""

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# Context variable for session-scoped logging
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


class LogContext:
    """Context manager for session-scoped logging."""

    def __init__(self, **kwargs):
        self._context = kwargs
        self._token = None

    def __enter__(self):
        current = _log_context.get().copy()
        current.update(self._context)
        self._token = _log_context.set(current)
        return self

    def __exit__(self, *args):
        if self._token:
            _log_context.reset(self._token)

    @staticmethod
    def get() -> dict[str, Any]:
        return _log_context.get()

    @staticmethod
    def set(**kwargs):
        current = _log_context.get().copy()
        current.update(kwargs)
        _log_context.set(current)


def add_context_processor(logger, method_name, event_dict):
    """Processor that adds context from ContextVar."""
    event_dict.update(_log_context.get())
    return event_dict


def setup_logging(level: str = "INFO", json_output: bool = False):
    """Configure structured logging."""

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_context_processor,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if json_output:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger, optionally with a name bound."""
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


# Default setup
setup_logging()
