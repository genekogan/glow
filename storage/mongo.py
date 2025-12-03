"""MongoDB storage implementation."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from models import Session, User, Event, Message, utcnow
from .base import Storage


class MongoStorage(Storage):
    """MongoDB storage backend using Motor (async driver)."""

    def __init__(self, uri: str, database: str):
        self.uri = uri
        self.database_name = database
        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._db

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    async def create_session(self, session: Session) -> Session:
        await self.db.sessions.insert_one(session.model_dump())
        return session

    async def get_session(self, session_id: str) -> Session | None:
        doc = await self.db.sessions.find_one({"id": session_id})
        return Session(**doc) if doc else None

    async def update_session(self, session: Session) -> Session:
        session.updated_at = utcnow()
        await self.db.sessions.replace_one(
            {"id": session.id},
            session.model_dump(),
            upsert=True,
        )
        return session

    async def delete_session(self, session_id: str) -> bool:
        result = await self.db.sessions.delete_one({"id": session_id})
        return result.deleted_count > 0

    async def list_sessions(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        query = {}
        if user_id:
            query["user_id"] = user_id
        if status:
            query["status"] = status

        cursor = (
            self.db.sessions.find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        return [Session(**doc) async for doc in cursor]

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================

    async def create_user(self, user: User) -> User:
        await self.db.users.insert_one(user.model_dump())
        return user

    async def get_user(self, user_id: str) -> User | None:
        doc = await self.db.users.find_one({"id": user_id})
        return User(**doc) if doc else None

    async def get_user_by_external_id(self, external_id: str) -> User | None:
        doc = await self.db.users.find_one({"external_id": external_id})
        return User(**doc) if doc else None

    async def update_user(self, user: User) -> User:
        user.updated_at = utcnow()
        await self.db.users.replace_one(
            {"id": user.id},
            user.model_dump(),
            upsert=True,
        )
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
        await self.db.sessions.update_one(
            {"id": session_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": utcnow()},
            },
        )
        return message

    async def append_event(self, session_id: str, event: Event) -> Event:
        await self.db.sessions.update_one(
            {"id": session_id},
            {
                "$push": {"events": event.model_dump()},
                "$set": {"updated_at": utcnow()},
            },
        )
        return event

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def connect(self):
        self._client = AsyncIOMotorClient(self.uri)
        self._db = self._client[self.database_name]

        # Create indexes
        await self.db.sessions.create_index("id", unique=True)
        await self.db.sessions.create_index("user_id")
        await self.db.sessions.create_index("status")
        await self.db.sessions.create_index("created_at")
        await self.db.sessions.create_index("anthropic_session_id")

        await self.db.users.create_index("id", unique=True)
        await self.db.users.create_index("external_id", unique=True, sparse=True)

    async def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
