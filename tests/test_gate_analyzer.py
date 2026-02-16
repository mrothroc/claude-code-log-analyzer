"""Tests for gate_analyzer.py pure functions.

Tests cover parsing, sanitization, and classification logic
that doesn't require API keys or real session data.
"""

import json
import pytest

from gate_analyzer import (
    _canonicalize_decision,
    _sanitize_path,
    _sanitize_project,
    parse_go_duration,
    parse_review_result,
)


# ── parse_go_duration ────────────────────────────────────────────────────

class TestParseGoDuration:
    def test_simple_seconds(self):
        assert parse_go_duration("5s") == 5.0

    def test_minutes_and_seconds(self):
        assert parse_go_duration("2m30s") == 150.0

    def test_hours_minutes_seconds(self):
        assert parse_go_duration("1h2m3s") == 3723.0

    def test_milliseconds_not_minutes(self):
        """The original bug: 75ms was parsed as 75 minutes."""
        result = parse_go_duration("75ms")
        assert result == pytest.approx(0.075)
        assert result < 1.0  # Definitely not 75 minutes

    def test_microseconds(self):
        assert parse_go_duration("500us") == pytest.approx(0.0005)

    def test_unicode_microseconds(self):
        assert parse_go_duration("500µs") == pytest.approx(0.0005)

    def test_nanoseconds(self):
        assert parse_go_duration("100ns") == pytest.approx(0.0000001)

    def test_complex_duration(self):
        # 1h30m500ms
        result = parse_go_duration("1h30m500ms")
        assert result == pytest.approx(5400.5)

    def test_none_input(self):
        assert parse_go_duration(None) is None

    def test_empty_string(self):
        assert parse_go_duration("") is None

    def test_non_string(self):
        assert parse_go_duration(12345) is None

    def test_no_units(self):
        assert parse_go_duration("hello") is None

    def test_fractional_seconds(self):
        assert parse_go_duration("1.5s") == pytest.approx(1.5)


# ── _canonicalize_decision ───────────────────────────────────────────────

class TestCanonicalizeDecision:
    @pytest.mark.parametrize("raw", [
        "APPROVED", "approved", "PASS", "PASSED", "LGTM",
        "SUCCESS", "ACCEPT", "ACCEPTED",
    ])
    def test_approved_variants(self, raw):
        assert _canonicalize_decision(raw) == "APPROVED"

    @pytest.mark.parametrize("raw", [
        "NEEDS_REVISION", "REJECTED", "REJECT", "FAILED", "FAIL",
        "NEEDS REVISION", "DENIED", "NOT_APPROVED",
    ])
    def test_needs_revision_variants(self, raw):
        assert _canonicalize_decision(raw) == "NEEDS_REVISION"

    @pytest.mark.parametrize("raw", ["ESCALATE", "ESCALATED", "ESCALATION"])
    def test_escalate_variants(self, raw):
        assert _canonicalize_decision(raw) == "ESCALATE"

    def test_unknown_returns_none(self):
        assert _canonicalize_decision("MAYBE") is None

    def test_whitespace_stripped(self):
        assert _canonicalize_decision("  APPROVED  ") == "APPROVED"

    def test_case_insensitive(self):
        assert _canonicalize_decision("Approved") == "APPROVED"
        assert _canonicalize_decision("rejected") == "NEEDS_REVISION"


# ── _sanitize_project ───────────────────────────────────────────────────

class TestSanitizeProject:
    def test_full_path_project_id(self):
        assert _sanitize_project("-Users-alice-IdeaProjects-myapp") == "myapp"

    def test_simple_name_unchanged(self):
        assert _sanitize_project("myapp") == "myapp"

    def test_none_returns_empty(self):
        assert _sanitize_project(None) == ""

    def test_empty_returns_empty(self):
        assert _sanitize_project("") == ""

    def test_single_segment(self):
        assert _sanitize_project("project") == "project"

    def test_linux_style_path(self):
        assert _sanitize_project("-home-user-code-webapp") == "webapp"

    def test_leading_dashes_stripped(self):
        # "-Users-bob-app" splits to ["Users", "bob", "app"]
        assert _sanitize_project("-Users-bob-app") == "app"


# ── _sanitize_path ──────────────────────────────────────────────────────

class TestSanitizePath:
    def test_replaces_home(self):
        from pathlib import Path
        home = str(Path.home())
        result = _sanitize_path(f"{home}/projects/foo")
        assert result == "~/projects/foo"
        assert home not in result

    def test_non_home_path_unchanged(self):
        assert _sanitize_path("/tmp/foo") == "/tmp/foo"


# ── parse_review_result ─────────────────────────────────────────────────

class TestParseReviewResult:
    def test_direct_decision_approved(self):
        data = {"decision": "APPROVED", "feedback": "Looks good"}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"
        assert result["feedback_text"] == "Looks good"

    def test_direct_decision_canonicalized(self):
        data = {"decision": "PASSED", "feedback": "All checks pass"}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

    def test_unrecognized_decision_falls_through(self):
        """Unrecognized decision value should not short-circuit."""
        data = {"decision": "MAYBE", "status": "APPROVED"}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

    def test_code_review_status_no_issues(self):
        data = {"code_review_status": {"issues_found": 0, "current_confidence": "high"}}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

    def test_code_review_status_with_issues_is_ambiguous(self):
        """Issues found should NOT default to APPROVED."""
        data = {"code_review_status": {"issues_found": 3, "current_confidence": "high"}}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] is None

    def test_workflow_guidance_passed(self):
        data = {"workflow_guidance": {"guidance": "All checks PASSED"}}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

    def test_workflow_guidance_failed(self):
        data = {"workflow_guidance": {"guidance": "FAILED: missing tests"}}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "NEEDS_REVISION"

    def test_disposition_tag(self):
        text = "Some review text <disposition>NEEDS_REVISION</disposition> more text"
        result = parse_review_result(text)
        assert result["decision"] == "NEEDS_REVISION"

    def test_non_json_returns_feedback_only(self):
        result = parse_review_result("This is plain text feedback")
        assert result["decision"] is None
        assert result["feedback_text"] == "This is plain text feedback"

    def test_empty_string(self):
        result = parse_review_result("")
        assert result["decision"] is None
        assert result["feedback_text"] is None

    def test_dict_feedback_normalized_to_string(self):
        """Non-string feedback should be JSON-serialized, not crash sqlite."""
        data = {"decision": "APPROVED", "feedback": {"detail": "all good", "score": 9}}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"
        assert isinstance(result["feedback_text"], str)
        assert "all good" in result["feedback_text"]

    def test_execution_time_parsed(self):
        data = {"decision": "APPROVED", "execution_time": "2m30s"}
        result = parse_review_result(json.dumps(data))
        assert result["execution_time_seconds"] == pytest.approx(150.0)
