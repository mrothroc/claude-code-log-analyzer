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
    _is_workflow_status_message,
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

    def test_two_phase_body_approval_overrides_stale_wrapper_rejection(self):
        feedback = """# Design Review Analysis

## Two-Phase Meta-Prompting Architecture

**Decision**: NEEDS_REVISION

## AI Analysis
**DECISION:** APPROVED

The design is approved without reservation.
"""
        data = {"decision": "NEEDS_REVISION", "feedback": feedback}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

    def test_two_phase_body_rejection_stays_rejected(self):
        feedback = """# Design Review Analysis

## Two-Phase Meta-Prompting Architecture

**Decision**: NEEDS_REVISION

## AI Analysis
**DECISION:** NEEDS_REVISION

The design requires revision before implementation.
"""
        data = {"decision": "NEEDS_REVISION", "feedback": feedback}
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "NEEDS_REVISION"

    def test_plain_rejection_approved_phrase_does_not_override_disposition(self):
        text = """<disposition>NEEDS_REVISION</disposition>

The proposed design requires revision before it can be approved for implementation.
"""
        result = parse_review_result(text)
        assert result["decision"] == "NEEDS_REVISION"

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

    def test_workflow_status_messages_parsed_as_status_only(self):
        # 1. codereview row with status: "code_review_complete_ready_for_implementation" + next_step_required: false -> STATUS_ONLY
        data1 = {"status": "code_review_complete_ready_for_implementation", "next_step_required": False}
        result1 = parse_review_result(json.dumps(data1))
        assert result1["decision"] == "STATUS_ONLY"

        # 2. codereview row with next_step_required: true, no findings -> STATUS_ONLY
        data2 = {"status": "code_review_complete_ready_for_implementation", "next_step_required": True}
        result2 = parse_review_result(json.dumps(data2))
        assert result2["decision"] == "STATUS_ONLY"

    def test_real_verdict_with_findings_retains_classification(self):
        # Real verdict + real issues_found -> classify correctly (no regression)
        data = {
            "status": "complete",
            "decision": "APPROVED",
            "issues_found": 0,
            "feedback": "All looks great"
        }
        result = parse_review_result(json.dumps(data))
        assert result["decision"] == "APPROVED"

        data_fail = {
            "status": "complete",
            "decision": "NEEDS_REVISION",
            "issues_found": 2,
            "feedback": "Requires fixing unit tests"
        }
        result_fail = parse_review_result(json.dumps(data_fail))
        assert result_fail["decision"] == "NEEDS_REVISION"


# ── _is_workflow_status_message ──────────────────────────────────────────

class TestIsWorkflowStatusMessage:
    def test_valid_status_messages(self):
        # 1. status: "code_review_complete_ready_for_implementation" + next_step_required: false
        data1 = {"status": "code_review_complete_ready_for_implementation", "next_step_required": False}
        assert _is_workflow_status_message(json.dumps(data1)) is True

        # 2. status: "code_review_paused_needs_revision" + next_step_required: true (no findings or decision)
        data2 = {"status": "code_review_paused_needs_revision", "next_step_required": True}
        assert _is_workflow_status_message(json.dumps(data2)) is True

        # 3. status matching continued with step_number
        data3 = {"status": "code_review_continued", "step_number": 2}
        assert _is_workflow_status_message(json.dumps(data3)) is True

    def test_invalid_status_messages(self):
        # Lacks step keys
        data = {"status": "code_review_complete_ready_for_implementation"}
        assert _is_workflow_status_message(json.dumps(data)) is False

        # Non-matching status value
        data = {"status": "unknown_status", "next_step_required": True}
        assert _is_workflow_status_message(json.dumps(data)) is False

        # Has explicit APPROVED verdict
        data = {
            "status": "code_review_complete_ready_for_implementation",
            "next_step_required": False,
            "decision": "APPROVED"
        }
        assert _is_workflow_status_message(json.dumps(data)) is False

        # Has non-empty issues_found
        data = {
            "status": "code_review_complete_ready_for_implementation",
            "next_step_required": True,
            "issues_found": 3
        }
        assert _is_workflow_status_message(json.dumps(data)) is False

        # Has non-empty findings list
        data = {
            "status": "code_review_complete_ready_for_implementation",
            "next_step_required": True,
            "findings": ["issue 1"]
        }
        assert _is_workflow_status_message(json.dumps(data)) is False

