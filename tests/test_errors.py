"""Tests for error handling."""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from errors import (
    AgentError, RetryableError, AuthError, BadRequestError,
    ModerationError, RateLimitError, OverloadedError, CancelledError,
    ErrorCategory, classify_anthropic_error, is_retryable, with_retry,
)


class TestErrorTypes:
    def test_agent_error(self):
        err = AgentError("Something went wrong")
        assert str(err) == "[unknown] Something went wrong"
        assert err.category == ErrorCategory.UNKNOWN

    def test_retryable_error(self):
        err = RetryableError("Server error", status_code=500)
        assert err.category == ErrorCategory.RETRYABLE
        assert err.status_code == 500

    def test_auth_error(self):
        err = AuthError("Invalid API key")
        assert err.category == ErrorCategory.AUTH

    def test_bad_request_error(self):
        err = BadRequestError("Invalid parameter")
        assert err.category == ErrorCategory.BAD_REQUEST

    def test_moderation_error(self):
        err = ModerationError("Content policy violation")
        assert err.category == ErrorCategory.MODERATION

    def test_rate_limit_error(self):
        err = RateLimitError("Too many requests", retry_after=30.0)
        assert err.category == ErrorCategory.RATE_LIMIT
        assert err.retry_after == 30.0

    def test_overloaded_error(self):
        err = OverloadedError("Server overloaded")
        assert err.category == ErrorCategory.OVERLOADED

    def test_cancelled_error(self):
        err = CancelledError()
        assert err.category == ErrorCategory.CANCELLED
        assert "cancelled" in str(err).lower()

    def test_error_with_original(self):
        original = ValueError("Original error")
        err = AgentError("Wrapped error", original=original)
        assert err.original == original


class TestErrorClassification:
    def test_classify_asyncio_cancelled(self):
        err = asyncio.CancelledError()
        classified = classify_anthropic_error(err)
        assert isinstance(classified, CancelledError)

    def test_classify_unknown_error(self):
        err = ValueError("Unknown")
        classified = classify_anthropic_error(err)
        assert isinstance(classified, AgentError)
        assert classified.category == ErrorCategory.UNKNOWN


class TestIsRetryable:
    def test_retryable_errors_are_retryable(self):
        assert is_retryable(RetryableError("test"))
        assert is_retryable(RateLimitError("test"))
        assert is_retryable(OverloadedError("test"))

    def test_non_retryable_errors(self):
        assert not is_retryable(AuthError("test"))
        assert not is_retryable(BadRequestError("test"))
        assert not is_retryable(ModerationError("test"))
        assert not is_retryable(CancelledError())


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        call_count = 0

        @with_retry(max_attempts=3)
        async def successful():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.01)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure")
            return "success"

        result = await flaky()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        call_count = 0

        @with_retry(max_attempts=3)
        async def auth_failure():
            nonlocal call_count
            call_count += 1
            raise AuthError("Invalid key")

        with pytest.raises(AuthError):
            await auth_failure()

        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Always fails")

        with pytest.raises(RetryableError):
            await always_fails()

        assert call_count == 3
