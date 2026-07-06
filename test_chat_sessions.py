#!/usr/bin/env python3
"""
Tests for chat session memory (ChatStore).

Covers:
- Creating sessions
- Listing sessions (newest first)
- Getting a session with full message history
- Adding messages (user + assistant)
- Auto-titling from first user message
- Deleting a session
- Max 10 session limit (oldest evicted automatically)
- Persistence across ChatStore instances (SQLite)

Run with:
    pytest test_chat_sessions.py -v
Or as a standalone script:
    python test_chat_sessions.py
"""

import sys
import tempfile
from pathlib import Path

import pytest

from app.chat.chat_store import ChatStore, MAX_SESSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path) -> ChatStore:
    """Return a ChatStore backed by a temp SQLite file."""
    return ChatStore(db_path=tmp_path / "test_sessions.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_returns_session_with_id(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        assert session.session_id
        assert len(session.session_id) == 36  # UUID format

    def test_default_title_uses_date(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        assert session.title.startswith("Chat ")

    def test_custom_title_is_stored(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session(title="My FastAPI session")
        assert session.title == "My FastAPI session"

    def test_new_session_has_no_messages(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        assert session.messages == []

    def test_created_at_and_updated_at_are_set(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        assert session.created_at
        assert session.updated_at


class TestGetSession:
    def test_returns_none_for_missing_id(self, tmp_path):
        store = make_store(tmp_path)
        result = store.get_session("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_returns_correct_session(self, tmp_path):
        store = make_store(tmp_path)
        created = store.create_session(title="Test")
        fetched = store.get_session(created.session_id)
        assert fetched is not None
        assert fetched.session_id == created.session_id
        assert fetched.title == "Test"

    def test_messages_are_returned_in_order(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        store.add_message(session.session_id, "user", "first")
        store.add_message(session.session_id, "assistant", "second")
        store.add_message(session.session_id, "user", "third")

        fetched = store.get_session(session.session_id)
        assert [m.content for m in fetched.messages] == ["first", "second", "third"]
        assert [m.role for m in fetched.messages] == ["user", "assistant", "user"]


class TestListSessions:
    def test_empty_store_returns_empty_list(self, tmp_path):
        store = make_store(tmp_path)
        assert store.list_sessions() == []

    def test_sessions_returned_newest_first(self, tmp_path):
        store = make_store(tmp_path)
        s1 = store.create_session(title="First")
        s2 = store.create_session(title="Second")
        s3 = store.create_session(title="Third")

        listed = store.list_sessions()
        # Most recently created should come first
        assert listed[0].session_id == s3.session_id
        assert listed[-1].session_id == s1.session_id

    def test_list_does_not_include_messages(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        store.add_message(session.session_id, "user", "hello")

        listed = store.list_sessions()
        assert listed[0].messages == []  # Messages not loaded in list view


class TestAddMessage:
    def test_message_is_persisted(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        result = store.add_message(session.session_id, "user", "What is RAG?")
        assert result is True

        fetched = store.get_session(session.session_id)
        assert len(fetched.messages) == 1
        assert fetched.messages[0].content == "What is RAG?"
        assert fetched.messages[0].role == "user"

    def test_returns_false_for_missing_session(self, tmp_path):
        store = make_store(tmp_path)
        result = store.add_message(
            "00000000-0000-0000-0000-000000000000", "user", "hello"
        )
        assert result is False

    def test_auto_title_from_first_user_message(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()  # Gets default "Chat ..." title
        assert session.title.startswith("Chat ")

        store.add_message(session.session_id, "user", "How does dependency injection work?")
        fetched = store.get_session(session.session_id)
        assert fetched.title == "How does dependency injection work?"

    def test_auto_title_truncated_at_60_chars(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        long_question = "A" * 80
        store.add_message(session.session_id, "user", long_question)

        fetched = store.get_session(session.session_id)
        assert len(fetched.title) <= 63  # 60 chars + "..."

    def test_assistant_message_does_not_change_title(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session(title="My custom title")
        store.add_message(session.session_id, "assistant", "Here is the answer.")

        fetched = store.get_session(session.session_id)
        assert fetched.title == "My custom title"

    def test_updated_at_changes_after_message(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        original_updated = session.updated_at

        store.add_message(session.session_id, "user", "hello")
        fetched = store.get_session(session.session_id)
        assert fetched.updated_at >= original_updated


class TestDeleteSession:
    def test_deletes_existing_session(self, tmp_path):
        store = make_store(tmp_path)
        session = store.create_session()
        result = store.delete_session(session.session_id)
        assert result is True
        assert store.get_session(session.session_id) is None

    def test_returns_false_for_missing_session(self, tmp_path):
        store = make_store(tmp_path)
        result = store.delete_session("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_count_decreases_after_delete(self, tmp_path):
        store = make_store(tmp_path)
        s1 = store.create_session()
        s2 = store.create_session()
        assert store.session_count() == 2

        store.delete_session(s1.session_id)
        assert store.session_count() == 1


class TestSessionLimit:
    def test_enforces_max_10_sessions(self, tmp_path):
        store = make_store(tmp_path)
        for i in range(MAX_SESSIONS + 3):
            store.create_session(title=f"Session {i}")

        assert store.session_count() == MAX_SESSIONS

    def test_oldest_session_evicted(self, tmp_path):
        store = make_store(tmp_path)
        first = store.create_session(title="Oldest")
        for i in range(MAX_SESSIONS - 1):
            store.create_session(title=f"Session {i}")

        # Now create one more — should evict "Oldest"
        store.create_session(title="New one")

        assert store.get_session(first.session_id) is None
        assert store.session_count() == MAX_SESSIONS


class TestPersistence:
    def test_session_survives_store_reinitialisation(self, tmp_path):
        """Data must persist across ChatStore instances (SQLite file)."""
        db_path = tmp_path / "sessions.db"

        store1 = ChatStore(db_path=db_path)
        session = store1.create_session(title="Persistent session")
        store1.add_message(session.session_id, "user", "hello")

        # Simulate application restart
        store2 = ChatStore(db_path=db_path)
        fetched = store2.get_session(session.session_id)

        assert fetched is not None
        assert fetched.title == "Persistent session"
        assert len(fetched.messages) == 1
        assert fetched.messages[0].content == "hello"


# ---------------------------------------------------------------------------
# Standalone demo (keeps original project style)
# ---------------------------------------------------------------------------

def demo():
    """Demo run showing the full chat session workflow."""
    import tempfile

    print("=" * 70)
    print("CHAT SESSION STORE DEMO")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        store = ChatStore(db_path=Path(tmp) / "demo.db")

        # Create session
        session = store.create_session()
        print(f"\n✅ Created session: {session.session_id}")
        print(f"   Title: {session.title}")

        # Add messages
        store.add_message(session.session_id, "user", "How does FastAPI handle dependencies?")
        store.add_message(session.session_id, "assistant", "FastAPI uses Depends() for DI...")
        store.add_message(session.session_id, "user", "Can you show me an example?")
        store.add_message(session.session_id, "assistant", "Sure! Here is an example...")
        print(f"\n✅ Added 4 messages")

        # Auto-title check
        fetched = store.get_session(session.session_id)
        print(f"   Auto-title: '{fetched.title}'")

        # List sessions
        sessions = store.list_sessions()
        print(f"\n✅ Listed {len(sessions)} session(s)")

        # Get full history
        full = store.get_session(session.session_id)
        print(f"\n✅ Full history ({len(full.messages)} messages):")
        for msg in full.messages:
            print(f"   [{msg.role}] {msg.content[:60]}")

        # Session limit demo
        print(f"\n⏳ Creating {MAX_SESSIONS + 2} sessions to test limit...")
        for i in range(MAX_SESSIONS + 2):
            store.create_session(title=f"Session {i}")
        print(f"✅ Session count capped at: {store.session_count()} (limit={MAX_SESSIONS})")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
