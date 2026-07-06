#!/usr/bin/env python3
"""
Tests for input sanitization validators in API models.

Covers:
- IndexRepositoryRequest: URL validation, branch validation
- ChatRequest: question sanitization, session_id UUID check,
               repository_name path traversal, language validation,
               max_tokens / temperature bounds

Run with:
    pytest test_models_validation.py -v
Or as a standalone script:
    python test_models_validation.py
"""

import sys
import pytest
from pydantic import ValidationError

from app.api.models import IndexRepositoryRequest, ChatRequest


# ---------------------------------------------------------------------------
# IndexRepositoryRequest — repository_url
# ---------------------------------------------------------------------------

class TestRepositoryUrlValidation:
    def test_valid_https_git_url(self):
        req = IndexRepositoryRequest(
            repository_url="https://github.com/fastapi/fastapi.git"
        )
        assert req.repository_url == "https://github.com/fastapi/fastapi.git"

    def test_valid_ssh_url(self):
        req = IndexRepositoryRequest(
            repository_url="git@github.com:fastapi/fastapi.git"
        )
        assert req.repository_url == "git@github.com:fastapi/fastapi.git"

    def test_valid_gitlab_url(self):
        req = IndexRepositoryRequest(
            repository_url="https://gitlab.com/user/repo.git"
        )
        assert req.repository_url

    def test_invalid_plain_string_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            IndexRepositoryRequest(repository_url="not-a-url")
        assert "Invalid Git URL" in str(exc_info.value)

    def test_file_protocol_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            IndexRepositoryRequest(repository_url="file:///etc/passwd")
        assert "path traversal" in str(exc_info.value).lower() or \
               "Invalid Git URL" in str(exc_info.value)

    def test_path_traversal_in_url_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            IndexRepositoryRequest(
                repository_url="https://github.com/../../../etc/passwd"
            )
        assert "path traversal" in str(exc_info.value).lower()

    def test_empty_url_rejected(self):
        with pytest.raises(ValidationError):
            IndexRepositoryRequest(repository_url="")


# ---------------------------------------------------------------------------
# IndexRepositoryRequest — branch
# ---------------------------------------------------------------------------

class TestBranchValidation:
    def test_valid_branch_name(self):
        req = IndexRepositoryRequest(
            repository_url="https://github.com/fastapi/fastapi.git",
            branch="main"
        )
        assert req.branch == "main"

    def test_valid_branch_with_slash(self):
        req = IndexRepositoryRequest(
            repository_url="https://github.com/fastapi/fastapi.git",
            branch="feature/new-api"
        )
        assert req.branch == "feature/new-api"

    def test_none_branch_allowed(self):
        req = IndexRepositoryRequest(
            repository_url="https://github.com/fastapi/fastapi.git",
            branch=None
        )
        assert req.branch is None

    def test_semicolon_in_branch_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            IndexRepositoryRequest(
                repository_url="https://github.com/fastapi/fastapi.git",
                branch="main; rm -rf /"
            )
        assert "invalid" in str(exc_info.value).lower()

    def test_pipe_in_branch_rejected(self):
        with pytest.raises(ValidationError):
            IndexRepositoryRequest(
                repository_url="https://github.com/fastapi/fastapi.git",
                branch="main|cat /etc/passwd"
            )

    def test_backtick_in_branch_rejected(self):
        with pytest.raises(ValidationError):
            IndexRepositoryRequest(
                repository_url="https://github.com/fastapi/fastapi.git",
                branch="`whoami`"
            )

    def test_newline_in_branch_rejected(self):
        with pytest.raises(ValidationError):
            IndexRepositoryRequest(
                repository_url="https://github.com/fastapi/fastapi.git",
                branch="main\nrm -rf /"
            )


# ---------------------------------------------------------------------------
# ChatRequest — question
# ---------------------------------------------------------------------------

class TestQuestionValidation:
    def test_valid_question(self):
        req = ChatRequest(question="How does FastAPI handle dependencies?")
        assert req.question == "How does FastAPI handle dependencies?"

    def test_whitespace_is_normalized(self):
        req = ChatRequest(question="  How   does   FastAPI work?  ")
        assert req.question == "How does FastAPI work?"

    def test_question_too_short_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="Hi")

    def test_question_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="A" * 2001)

    def test_excessive_repetition_rejected(self):
        # >70% of words are the same
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question="word " * 60)
        assert "repetition" in str(exc_info.value).lower()

    def test_normal_repetition_ok(self):
        # A real question that happens to repeat a few words
        req = ChatRequest(question="What is the difference between the old API and the new API?")
        assert req.question


# ---------------------------------------------------------------------------
# ChatRequest — session_id
# ---------------------------------------------------------------------------

class TestSessionIdValidation:
    def test_valid_uuid(self):
        req = ChatRequest(
            question="How does this work?",
            session_id="123e4567-e89b-12d3-a456-426614174000"
        )
        assert req.session_id == "123e4567-e89b-12d3-a456-426614174000"

    def test_none_session_id_allowed(self):
        req = ChatRequest(question="How does this work?", session_id=None)
        assert req.session_id is None

    def test_invalid_uuid_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question="How does this work?", session_id="not-a-uuid")
        assert "UUID" in str(exc_info.value)

    def test_uuid_without_dashes_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                question="How does this work?",
                session_id="123e4567e89b12d3a456426614174000"
            )


# ---------------------------------------------------------------------------
# ChatRequest — repository_name
# ---------------------------------------------------------------------------

class TestRepositoryNameValidation:
    def test_valid_repo_name(self):
        req = ChatRequest(question="How does this work?", repository_name="fastapi")
        assert req.repository_name == "fastapi"

    def test_none_allowed(self):
        req = ChatRequest(question="How does this work?", repository_name=None)
        assert req.repository_name is None

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question="How does this work?", repository_name="../etc/passwd")
        assert "path traversal" in str(exc_info.value).lower()

    def test_absolute_path_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", repository_name="/etc/passwd")

    def test_windows_path_traversal_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", repository_name="..\\secrets")


# ---------------------------------------------------------------------------
# ChatRequest — language
# ---------------------------------------------------------------------------

class TestLanguageValidation:
    def test_valid_language(self):
        req = ChatRequest(question="How does this work?", language="python")
        assert req.language == "python"

    def test_language_lowercased(self):
        req = ChatRequest(question="How does this work?", language="Python")
        assert req.language == "python"

    def test_none_allowed(self):
        req = ChatRequest(question="How does this work?", language=None)
        assert req.language is None

    def test_invalid_language_with_spaces_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", language="python; drop table")

    def test_language_with_dash_ok(self):
        req = ChatRequest(question="How does this work?", language="c-sharp")
        assert req.language == "c-sharp"


# ---------------------------------------------------------------------------
# ChatRequest — max_tokens / temperature bounds
# ---------------------------------------------------------------------------

class TestBoundaryValidation:
    def test_max_tokens_default(self):
        req = ChatRequest(question="How does this work?")
        assert req.max_tokens == 512

    def test_max_tokens_upper_bound(self):
        req = ChatRequest(question="How does this work?", max_tokens=4096)
        assert req.max_tokens == 4096

    def test_max_tokens_above_limit_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", max_tokens=4097)

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", max_tokens=0)

    def test_temperature_default(self):
        req = ChatRequest(question="How does this work?")
        assert req.temperature == 0.1

    def test_temperature_bounds_valid(self):
        req = ChatRequest(question="How does this work?", temperature=0.0)
        assert req.temperature == 0.0
        req2 = ChatRequest(question="How does this work?", temperature=1.0)
        assert req2.temperature == 1.0

    def test_temperature_above_1_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", temperature=1.1)

    def test_temperature_below_0_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="How does this work?", temperature=-0.1)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def demo():
    """Demo showing validation in action."""
    print("=" * 70)
    print("INPUT VALIDATION DEMO")
    print("=" * 70)

    tests = [
        # (description, should_pass, model_cls, kwargs)
        ("Valid GitHub URL", True, IndexRepositoryRequest,
         {"repository_url": "https://github.com/fastapi/fastapi.git"}),
        ("Invalid URL (plain string)", False, IndexRepositoryRequest,
         {"repository_url": "not-a-url"}),
        ("Path traversal in URL", False, IndexRepositoryRequest,
         {"repository_url": "https://github.com/../../../etc/passwd"}),
        ("Shell injection in branch", False, IndexRepositoryRequest,
         {"repository_url": "https://github.com/fastapi/fastapi.git",
          "branch": "main; rm -rf /"}),
        ("Valid chat question", True, ChatRequest,
         {"question": "How does dependency injection work?"}),
        ("Question too short", False, ChatRequest,
         {"question": "Hi"}),
        ("Valid UUID session_id", True, ChatRequest,
         {"question": "How does this work?",
          "session_id": "123e4567-e89b-12d3-a456-426614174000"}),
        ("Invalid session_id", False, ChatRequest,
         {"question": "How does this work?", "session_id": "not-a-uuid"}),
        ("Path traversal in repo_name", False, ChatRequest,
         {"question": "How does this work?", "repository_name": "../secrets"}),
        ("max_tokens too high", False, ChatRequest,
         {"question": "How does this work?", "max_tokens": 9999}),
    ]

    passed = 0
    for desc, should_pass, model_cls, kwargs in tests:
        try:
            model_cls(**kwargs)
            result = "✅ PASS (accepted)" if should_pass else "❌ FAIL (should have rejected)"
            if should_pass:
                passed += 1
        except ValidationError as e:
            result = "❌ FAIL (should have accepted)" if should_pass else "✅ PASS (rejected correctly)"
            if not should_pass:
                passed += 1

        print(f"  {result}: {desc}")

    print(f"\n{'=' * 70}")
    print(f"Passed: {passed}/{len(tests)}")
    if passed == len(tests):
        print("✅ ALL VALIDATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")


if __name__ == "__main__":
    demo()
