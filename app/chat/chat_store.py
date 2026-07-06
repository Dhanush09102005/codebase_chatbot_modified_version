"""
Chat session store using SQLite.

Stores up to MAX_SESSIONS chat sessions. When the limit is reached,
the oldest session is automatically deleted to make room.

Each session holds:
- A unique ID and title (auto-generated from the first question)
- An ordered list of messages (role: user/assistant, content, timestamp)
- Created/updated timestamps

No external dependencies — uses Python's built-in sqlite3.
"""
import sqlite3
import uuid
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

from app.core import logger, settings

MAX_SESSIONS = 10


@dataclass
class ChatMessage:
    """A single message in a chat session."""
    role: str          # "user" or "assistant"
    content: str       # Message text
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(**data)


@dataclass
class ChatSession:
    """A chat session with its full message history."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }


class ChatStore:
    """
    SQLite-backed chat session store.

    Manages creation, retrieval, update, and deletion of chat sessions.
    Automatically enforces a maximum of MAX_SESSIONS (10) sessions by
    evicting the oldest session when the limit is exceeded.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.chroma_persist_directory.parent / "chat_sessions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"ChatStore initialized: {self.db_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.

        If creating this session would exceed MAX_SESSIONS, the oldest
        session (by created_at) is deleted first.

        Args:
            title: Optional title. Defaults to "Chat <date>".

        Returns:
            ChatSession: The newly created session.
        """
        self._enforce_session_limit()

        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        title = title or f"Chat {datetime.utcnow().strftime('%b %d, %H:%M')}"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at, messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, title, now, now, json.dumps([]))
            )

        session = ChatSession(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=[]
        )
        logger.info(f"Created session: {session_id} ({title})")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a session by ID, including all messages.

        Args:
            session_id: Session UUID.

        Returns:
            ChatSession or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, title, created_at, updated_at, messages "
                "FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

        if not row:
            return None

        messages = [ChatMessage.from_dict(m) for m in json.loads(row[4])]
        return ChatSession(
            session_id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
            messages=messages
        )

    def list_sessions(self) -> List[ChatSession]:
        """
        List all sessions ordered by most recently updated, without messages.

        Returns:
            List[ChatSession]: Sessions with empty messages list (summary only).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()

        return [
            ChatSession(
                session_id=row[0],
                title=row[1],
                created_at=row[2],
                updated_at=row[3],
                messages=[]  # Not loaded for list view
            )
            for row in rows
        ]

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Append a message to a session and update its timestamp.

        Also auto-titles the session from the first user message if the
        current title starts with "Chat " (the default).

        Args:
            session_id: Session UUID.
            role: "user" or "assistant".
            content: Message text.

        Returns:
            bool: True if session was found and message was added.
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        message = ChatMessage(role=role, content=content)
        session.messages.append(message)

        # Auto-title from first user message
        if role == "user" and session.title.startswith("Chat ") and len(session.messages) == 1:
            session.title = content[:60] + ("..." if len(content) > 60 else "")

        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET messages = ?, updated_at = ?, title = ? "
                "WHERE session_id = ?",
                (json.dumps([m.to_dict() for m in session.messages]), now, session.title, session_id)
            )

        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id: Session UUID.

        Returns:
            bool: True if a session was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted session: {session_id}")
        return deleted

    def session_count(self) -> int:
        """Return the number of stored sessions."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the sessions table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages   TEXT NOT NULL DEFAULT '[]'
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with row factory set."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _enforce_session_limit(self):
        """
        If we already have MAX_SESSIONS, delete the oldest one to make room.
        """
        if self.session_count() >= MAX_SESSIONS:
            with self._connect() as conn:
                oldest = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                if oldest:
                    conn.execute(
                        "DELETE FROM sessions WHERE session_id = ?", (oldest[0],)
                    )
                    logger.info(f"Evicted oldest session to stay under limit: {oldest[0]}")
