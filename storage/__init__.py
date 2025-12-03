"""Storage layer for sessions and users."""

from .base import Storage
from .memory import MemoryStorage

__all__ = ["Storage", "MemoryStorage"]

# Lazy import for mongo to avoid requiring motor when not used
def get_mongo_storage():
    from .mongo import MongoStorage
    return MongoStorage
