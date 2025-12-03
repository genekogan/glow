"""Abstract storage interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from models import Session, User, Event, Message


class Storage(ABC):
    """Abstract base class for storage backends."""

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    @abstractmethod
    async def create_session(self, session: Session) -> Session:
        """Create a new session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        pass

    @abstractmethod
    async def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions with optional filters."""
        pass

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================

    @abstractmethod
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        pass

    @abstractmethod
    async def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        pass

    @abstractmethod
    async def get_user_by_external_id(self, external_id: str) -> User | None:
        """Get a user by external auth ID."""
        pass

    @abstractmethod
    async def update_user(self, user: User) -> User:
        """Update an existing user."""
        pass

    @abstractmethod
    async def get_or_create_user(self, external_id: str, **defaults) -> tuple[User, bool]:
        """Get user by external_id or create with defaults. Returns (user, created)."""
        pass

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    @abstractmethod
    async def append_message(self, session_id: str, message: Message) -> Message:
        """Append a message to a session."""
        pass

    @abstractmethod
    async def append_event(self, session_id: str, event: Event) -> Event:
        """Append an event to a session."""
        pass

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    @abstractmethod
    async def connect(self):
        """Connect to the storage backend."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the storage backend."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
