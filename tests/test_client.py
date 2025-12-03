"""Tests for agent client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from client import AgentClient, CancellationToken, create_client
from storage import MemoryStorage
from models import Session, SessionStatus, MessageRole


class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert not token.is_cancelled()

    def test_cancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()

    def test_check_raises_when_cancelled(self):
        from errors import CancelledError
        token = CancellationToken()
        token.cancel()

        with pytest.raises(CancelledError):
            token.check()

    def test_check_passes_when_not_cancelled(self):
        token = CancellationToken()
        token.check()  # Should not raise

    @pytest.mark.asyncio
    async def test_wait(self):
        token = CancellationToken()

        async def cancel_after_delay():
            await asyncio.sleep(0.01)
            token.cancel()

        task = asyncio.create_task(cancel_after_delay())
        await token.wait()
        assert token.is_cancelled()
        await task

    @pytest.mark.asyncio
    async def test_register_and_cancel_task(self):
        token = CancellationToken()

        async def dummy():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(dummy())
        token.register_task(task)
        token.cancel()

        # Give the cancellation time to propagate
        await asyncio.sleep(0.01)
        assert task.cancelled()


class TestAgentClient:
    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    @pytest.fixture
    def client(self, storage):
        return AgentClient(storage=storage)

    @pytest.mark.asyncio
    async def test_create_session(self, client, storage):
        async with client:
            session = await client.create_session(
                user_id="user123",
                system_prompt="Test prompt",
                cwd="/tmp",
            )

            assert session.id
            assert session.user_id == "user123"
            assert session.system_prompt == "Test prompt"
            assert session.cwd == "/tmp"
            assert session.status == SessionStatus.ACTIVE

            # Should be persisted
            stored = await storage.get_session(session.id)
            assert stored is not None

    @pytest.mark.asyncio
    async def test_get_session(self, client, storage):
        async with client:
            created = await client.create_session()
            retrieved = await client.get_session(created.id)

            assert retrieved is not None
            assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_resume_session(self, client, storage):
        async with client:
            session = await client.create_session()
            session.complete(SessionStatus.COMPLETED)
            await storage.update_session(session)

            resumed = await client.resume_session(session.id)
            assert resumed is not None
            assert resumed.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_or_create_user_creates(self, client):
        async with client:
            user = await client.get_or_create_user(
                external_id="auth0|123",
                name="Test User",
                avatar_url="https://example.com/avatar.png",
            )

            assert user.external_id == "auth0|123"
            assert user.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_or_create_user_gets(self, client):
        async with client:
            user1 = await client.get_or_create_user(
                external_id="auth0|123",
                name="Original Name",
            )

            user2 = await client.get_or_create_user(
                external_id="auth0|123",
                name="Different Name",
            )

            assert user1.id == user2.id
            assert user2.name == "Original Name"


class TestCreateClient:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with create_client() as client:
            assert isinstance(client, AgentClient)
            session = await client.create_session()
            assert session is not None

    @pytest.mark.asyncio
    async def test_with_custom_storage(self):
        storage = MemoryStorage()
        async with create_client(storage=storage) as client:
            await client.create_session()
            sessions = await storage.list_sessions()
            assert len(sessions) == 1
