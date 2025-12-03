"""Tests for agent tools and handlers."""

import pytest
from agent import calculator_handler, web_search_handler


class TestCalculatorHandler:
    @pytest.mark.asyncio
    async def test_add(self):
        result = await calculator_handler({"operation": "add", "a": 2, "b": 3})
        assert "5" in result

    @pytest.mark.asyncio
    async def test_subtract(self):
        result = await calculator_handler({"operation": "subtract", "a": 10, "b": 4})
        assert "6" in result

    @pytest.mark.asyncio
    async def test_multiply(self):
        result = await calculator_handler({"operation": "multiply", "a": 7, "b": 8})
        assert "56" in result

    @pytest.mark.asyncio
    async def test_divide(self):
        result = await calculator_handler({"operation": "divide", "a": 10, "b": 2})
        assert "5" in result

    @pytest.mark.asyncio
    async def test_divide_by_zero(self):
        result = await calculator_handler({"operation": "divide", "a": 10, "b": 0})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        result = await calculator_handler({"operation": "add", "a": -5, "b": 3})
        assert "-2" in result

    @pytest.mark.asyncio
    async def test_floats(self):
        result = await calculator_handler({"operation": "multiply", "a": 2.5, "b": 4})
        assert "10" in result


class TestWebSearchHandler:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        result = await web_search_handler({"query": "python async"})
        assert "python async" in result
        assert "Results" in result

    @pytest.mark.asyncio
    async def test_search_with_special_chars(self):
        result = await web_search_handler({"query": "what is 2+2?"})
        assert "2+2" in result
