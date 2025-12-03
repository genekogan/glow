"""In-memory storage implementation for testing."""

from copy import deepcopy

from models import Session, User, Event, Message, utcnow
from .base import Storage


class MemoryStorage(Storage):
    """In-memory storage for testing and development."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._users: dict[str, User] = {}
        self._users_by_external_id: dict[str, str] = {}  # external_id -> user_id

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    async def create_session(self, session: Session) -> Session:
        self._sessions[session.id] = deepcopy(session)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        return deepcopy(session) if session else None

    async def update_session(self, session: Session) -> Session:
        session.updated_at = utcnow()
        self._sessions[session.id] = deepcopy(session)
        return session

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def list_sessions(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return [deepcopy(s) for s in sessions[offset : offset + limit]]

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================

    async def create_user(self, user: User) -> User:
        self._users[user.id] = deepcopy(user)
        if user.external_id:
            self._users_by_external_id[user.external_id] = user.id
        return user

    async def get_user(self, user_id: str) -> User | None:
        user = self._users.get(user_id)
        return deepcopy(user) if user else None

    async def get_user_by_external_id(self, external_id: str) -> User | None:
        user_id = self._users_by_external_id.get(external_id)
        if user_id:
            return await self.get_user(user_id)
        return None

    async def update_user(self, user: User) -> User:
        user.updated_at = utcnow()
        self._users[user.id] = deepcopy(user)
        if user.external_id:
            self._users_by_external_id[user.external_id] = user.id
        return user

    async def get_or_create_user(self, external_id: str, **defaults) -> tuple[User, bool]:
        existing = await self.get_user_by_external_id(external_id)
        if existing:
            return existing, False

        user = User(external_id=external_id, **defaults)
        await self.create_user(user)
        return user, True

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    async def append_message(self, session_id: str, message: Message) -> Message:
        session = self._sessions.get(session_id)
        if session:
            session.messages.append(deepcopy(message))
            session.updated_at = utcnow()
        return message

    async def append_event(self, session_id: str, event: Event) -> Event:
        session = self._sessions.get(session_id)
        if session:
            session.events.append(deepcopy(event))
            session.updated_at = utcnow()
        return event

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def connect(self):
        pass  # No-op for memory storage

    async def disconnect(self):
        pass  # No-op for memory storage

    def clear(self):
        """Clear all data (for testing)."""
        self._sessions.clear()
        self._users.clear()
        self._users_by_external_id.clear()
