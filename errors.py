"""Error handling and retry logic."""

from enum import Enum
from typing import Any, Callable, TypeVar
from functools import wraps
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState,
)

from config import settings
from observability.logging import get_logger

log = get_logger("errors")

T = TypeVar("T")


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

class ErrorCategory(str, Enum):
    """Categories of errors for different handling."""
    RETRYABLE = "retryable"          # 500, 529, 429 with backoff
    AUTH = "auth"                     # 401 - fatal, don't retry
    BAD_REQUEST = "bad_request"       # 400 - don't retry, fix request
    MODERATION = "moderation"         # Content policy - don't retry
    RATE_LIMIT = "rate_limit"         # 429 - retry with backoff
    OVERLOADED = "overloaded"         # 529 - retry with backoff
    INTERNAL = "internal"             # 500 - retry with backoff
    TIMEOUT = "timeout"               # Timeout - retry once
    CANCELLED = "cancelled"           # User cancelled - don't retry
    UNKNOWN = "unknown"               # Unknown - log and surface


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        original: Exception | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.original = original
        self.status_code = status_code
        self.retry_after = retry_after

    def __str__(self):
        return f"[{self.category.value}] {super().__str__()}"


class RetryableError(AgentError):
    """Error that should be retried."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RETRYABLE, **kwargs)


class AuthError(AgentError):
    """Authentication error - fatal."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTH, **kwargs)


class BadRequestError(AgentError):
    """Bad request - don't retry."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BAD_REQUEST, **kwargs)


class ModerationError(AgentError):
    """Content moderation error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODERATION, **kwargs)


class RateLimitError(RetryableError):
    """Rate limit hit."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.category = ErrorCategory.RATE_LIMIT


class OverloadedError(RetryableError):
    """Server overloaded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.category = ErrorCategory.OVERLOADED


class CancelledError(AgentError):
    """Operation cancelled by user."""

    def __init__(self, message: str = "Operation cancelled", **kwargs):
        super().__init__(message, category=ErrorCategory.CANCELLED, **kwargs)


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

def classify_anthropic_error(error: Exception) -> AgentError:
    """Classify an Anthropic SDK error into our error hierarchy."""
    try:
        import anthropic

        if isinstance(error, anthropic.AuthenticationError):
            return AuthError(str(error), original=error, status_code=401)

        if isinstance(error, anthropic.BadRequestError):
            # Check for moderation errors
            msg = str(error).lower()
            if "content" in msg and ("policy" in msg or "moderation" in msg or "safety" in msg):
                return ModerationError(str(error), original=error, status_code=400)
            return BadRequestError(str(error), original=error, status_code=400)

        if isinstance(error, anthropic.RateLimitError):
            # Try to extract retry-after
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after_header = error.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            return RateLimitError(str(error), retry_after=retry_after, original=error, status_code=429)

        if isinstance(error, anthropic.InternalServerError):
            return RetryableError(str(error), original=error, status_code=500)

        if isinstance(error, anthropic.APIStatusError):
            status = error.status_code
            if status == 529:
                return OverloadedError(str(error), original=error, status_code=529)
            if status >= 500:
                return RetryableError(str(error), original=error, status_code=status)
            return AgentError(str(error), original=error, status_code=status)

        if isinstance(error, anthropic.APITimeoutError):
            return RetryableError("Request timed out", original=error, category=ErrorCategory.TIMEOUT)

        if isinstance(error, anthropic.APIConnectionError):
            return RetryableError("Connection error", original=error)

    except ImportError:
        pass

    # Fallback for unknown errors
    if isinstance(error, asyncio.CancelledError):
        return CancelledError(original=error)

    return AgentError(str(error), original=error)


def is_retryable(error: Exception) -> bool:
    """Check if an error should be retried."""
    if isinstance(error, AgentError):
        return error.category in (
            ErrorCategory.RETRYABLE,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.OVERLOADED,
            ErrorCategory.INTERNAL,
        )

    classified = classify_anthropic_error(error)
    return is_retryable(classified)


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def log_retry(retry_state: RetryCallState):
    """Log retry attempts."""
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        log.warning(
            "retry_attempt",
            attempt=retry_state.attempt_number,
            error=str(exc),
            next_wait=retry_state.next_action.sleep if retry_state.next_action else None,
        )


def with_retry(
    max_attempts: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
    exponential_base: float | None = None,
):
    """
    Decorator for retrying async functions with exponential backoff.

    Uses settings from config.yaml by default.
    """
    cfg = settings.retry
    _max_attempts = max_attempts or cfg.max_attempts
    _initial_delay = initial_delay or cfg.initial_delay
    _max_delay = max_delay or cfg.max_delay
    _exp_base = exponential_base or cfg.exponential_base

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            @retry(
                stop=stop_after_attempt(_max_attempts),
                wait=wait_exponential(
                    multiplier=_initial_delay,
                    max=_max_delay,
                    exp_base=_exp_base,
                ),
                retry=retry_if_exception(is_retryable),
                before_sleep=log_retry,
                reraise=True,
            )
            async def _inner():
                try:
                    return await func(*args, **kwargs)
                except AgentError:
                    # Already classified, re-raise as-is
                    raise
                except Exception as e:
                    # Classify Anthropic SDK errors
                    classified = classify_anthropic_error(e)
                    if not is_retryable(classified):
                        raise classified from e
                    raise classified from e

            return await _inner()

        return wrapper

    return decorator
