"""Tests for storage layer."""

import pytest
from models import Session, Message, Event, User, MessageRole, EventType, SessionStatus
from storage import MemoryStorage


@pytest.fixture
def storage():
    return MemoryStorage()


class TestMemoryStorageSession:
    @pytest.mark.asyncio
    async def test_create_session(self, storage):
        session = Session(user_id="user1")
        created = await storage.create_session(session)
        assert created.id == session.id
        assert created.user_id == "user1"

    @pytest.mark.asyncio
    async def test_get_session(self, storage):
        session = Session(user_id="user1")
        await storage.create_session(session)

        retrieved = await storage.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, storage):
        retrieved = await storage.get_session("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_session(self, storage):
        session = Session()
        await storage.create_session(session)

        session.status = SessionStatus.COMPLETED
        session.turns = 5
        await storage.update_session(session)

        retrieved = await storage.get_session(session.id)
        assert retrieved.status == SessionStatus.COMPLETED
        assert retrieved.turns == 5

    @pytest.mark.asyncio
    async def test_delete_session(self, storage):
        session = Session()
        await storage.create_session(session)

        deleted = await storage.delete_session(session.id)
        assert deleted is True

        retrieved = await storage.get_session(session.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, storage):
        deleted = await storage.delete_session("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_sessions(self, storage):
        for i in range(5):
            await storage.create_session(Session(user_id=f"user{i}"))

        sessions = await storage.list_sessions()
        assert len(sessions) == 5

    @pytest.mark.asyncio
    async def test_list_sessions_by_user(self, storage):
        await storage.create_session(Session(user_id="user1"))
        await storage.create_session(Session(user_id="user1"))
        await storage.create_session(Session(user_id="user2"))

        sessions = await storage.list_sessions(user_id="user1")
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_by_status(self, storage):
        s1 = Session()
        s2 = Session()
        s3 = Session()
        s2.status = SessionStatus.COMPLETED
        s3.status = SessionStatus.COMPLETED

        await storage.create_session(s1)
        await storage.create_session(s2)
        await storage.create_session(s3)

        active = await storage.list_sessions(status=SessionStatus.ACTIVE)
        completed = await storage.list_sessions(status=SessionStatus.COMPLETED)

        assert len(active) == 1
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, storage):
        for i in range(10):
            await storage.create_session(Session())

        page1 = await storage.list_sessions(limit=3, offset=0)
        page2 = await storage.list_sessions(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id


class TestMemoryStorageUser:
    @pytest.mark.asyncio
    async def test_create_user(self, storage):
        user = User(name="Test User", external_id="ext123")
        created = await storage.create_user(user)
        assert created.id == user.id
        assert created.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_user(self, storage):
        user = User(name="Test User")
        await storage.create_user(user)

        retrieved = await storage.get_user(user.id)
        assert retrieved is not None
        assert retrieved.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_user_by_external_id(self, storage):
        user = User(name="Test User", external_id="auth0|123")
        await storage.create_user(user)

        retrieved = await storage.get_user_by_external_id("auth0|123")
        assert retrieved is not None
        assert retrieved.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_user_by_external_id_not_found(self, storage):
        retrieved = await storage.get_user_by_external_id("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_user(self, storage):
        user = User(name="Original Name")
        await storage.create_user(user)

        user.name = "Updated Name"
        user.avatar_url = "https://example.com/avatar.png"
        await storage.update_user(user)

        retrieved = await storage.get_user(user.id)
        assert retrieved.name == "Updated Name"
        assert retrieved.avatar_url == "https://example.com/avatar.png"

    @pytest.mark.asyncio
    async def test_get_or_create_user_creates(self, storage):
        user, created = await storage.get_or_create_user(
            external_id="new_user",
            name="New User",
            avatar_url="https://example.com/new.png",
        )
        assert created is True
        assert user.external_id == "new_user"
        assert user.name == "New User"

    @pytest.mark.asyncio
    async def test_get_or_create_user_gets(self, storage):
        # First create
        user1, created1 = await storage.get_or_create_user(
            external_id="existing_user",
            name="First Name",
        )
        assert created1 is True

        # Second get
        user2, created2 = await storage.get_or_create_user(
            external_id="existing_user",
            name="Different Name",  # Should be ignored
        )
        assert created2 is False
        assert user2.id == user1.id
        assert user2.name == "First Name"


class TestMemoryStorageMessages:
    @pytest.mark.asyncio
    async def test_append_message(self, storage):
        session = Session()
        await storage.create_session(session)

        msg = Message(role=MessageRole.USER, content="Hello")
        await storage.append_message(session.id, msg)

        retrieved = await storage.get_session(session.id)
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_append_multiple_messages(self, storage):
        session = Session()
        await storage.create_session(session)

        await storage.append_message(session.id, Message(role=MessageRole.USER, content="Hello"))
        await storage.append_message(session.id, Message(role=MessageRole.ASSISTANT, content="Hi!"))
        await storage.append_message(session.id, Message(role=MessageRole.USER, content="How are you?"))

        retrieved = await storage.get_session(session.id)
        assert len(retrieved.messages) == 3

    @pytest.mark.asyncio
    async def test_append_event(self, storage):
        session = Session()
        await storage.create_session(session)

        evt = Event(type=EventType.TURN_START, turn=1)
        await storage.append_event(session.id, evt)

        retrieved = await storage.get_session(session.id)
        assert len(retrieved.events) == 1
        assert retrieved.events[0].type == EventType.TURN_START


class TestMemoryStorageLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self, storage):
        async with storage:
            session = Session()
            await storage.create_session(session)
            retrieved = await storage.get_session(session.id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_clear(self, storage):
        await storage.create_session(Session())
        await storage.create_user(User(name="Test"))

        storage.clear()

        sessions = await storage.list_sessions()
        assert len(sessions) == 0


class TestMemoryStorageIsolation:
    @pytest.mark.asyncio
    async def test_session_isolation(self, storage):
        """Modifying retrieved session shouldn't affect stored session."""
        original = Session()
        original.turns = 5
        await storage.create_session(original)

        retrieved = await storage.get_session(original.id)
        retrieved.turns = 10

        stored = await storage.get_session(original.id)
        assert stored.turns == 5  # Should not be affected
