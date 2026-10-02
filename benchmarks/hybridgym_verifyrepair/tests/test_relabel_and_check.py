from typing import Any

from benchmarks.hybridgym_verifyrepair.reason_check import (
    apply_checks,
    gold_diff,
    hidden_test_sources,
    parse_reply,
)
from benchmarks.hybridgym_verifyrepair.relabel_candidates import relabel


def _cand(iid: str, resolved: bool) -> dict[str, Any]:
    return {"instance_id": iid, "candidate_patch": "p", "candidate_resolved": resolved}


def test_relabel_keeps_only_trustworthy_labels():
    candidates = [
        _cand("ok_correct", True),
        _cand("flaky", True),
        _cand("real_flaw", False),
        _cand("pre_existing_only", False),
        _cand("errored", False),
        _cand("not_graded", False),
    ]
    regrades = {
        "ok_correct": {"resolved": True},
        "flaky": {"resolved": False, "mismatched_tests": {"t": "FAILED/PASSED"}},
        "real_flaw": {
            "resolved": False,
            "mismatched_tests": {"t_issue": "FAILED/PASSED", "t_env": "PASSED/FAILED"},
        },
        "pre_existing_only": {
            "resolved": False,
            "mismatched_tests": {"t_env": "PASSED/FAILED", "t_c": "MISSING/FAILED"},
        },
        "errored": {"resolved": False, "error": "docker run failed"},
    }

    kept, dropped = relabel(candidates, regrades)

    assert [r["instance_id"] for r in kept] == ["ok_correct", "real_flaw"]
    assert kept[1]["failing_tests"] == ["t_issue"]
    assert dropped == {
        "label changed on re-grade": 1,
        "flawed only on tests expected to fail": 1,
        "re-grade error": 1,
        "not re-graded": 1,
    }


def test_parse_reply():
    raw = 'Thinking...\n```json\n{"evidence": "Yes", "reason": "no", "note": "n"}\n```'
    assert parse_reply(raw) == {"evidence": "yes", "reason": "no", "check_note": "n"}
    assert parse_reply("no json here")["evidence"] == "?"


def test_apply_checks_narrows_resolved_ids_and_resumes():
    report: dict[str, Any] = {
        "resolved_ids": ["a", "b", "c"],
        "results": [{"instance_id": i} for i in ("a", "b", "c", "d")],
    }
    apply_checks(
        report,
        {
            "a": {"evidence": "yes", "reason": "yes"},
            "b": {"evidence": "yes", "reason": "no"},
            "c": {"evidence": "?", "reason": "?"},
        },
    )
    assert report["resolved_ids"] == ["a"]
    assert report["verdict_correct_ids"] == ["a", "b", "c"]

    # A later run answers c; verdict_correct_ids stays the judge_eval list.
    apply_checks(report, {"c": {"evidence": "yes", "reason": "yes"}})
    assert report["resolved_ids"] == ["a", "c"]
    assert report["verdict_correct_ids"] == ["a", "b", "c"]


def test_hidden_test_sources_and_gold_diff():
    code = (
        "class TestX:\n    def test_a(self):\n        assert 1\n\n"
        "def test_b(loop):\n    assert 2\n"
    )
    src = hidden_test_sources([code], ["TestX.test_a", "test_b[pyloop]"])
    assert "# TestX.test_a" in src and "assert 1" in src and "assert 2" in src

    commit = {
        "file_diffs": [
            {
                "header": "{'file': {'path': 'pkg/mod.py'}}",
                "old_file_content": "a = 1\n",
                "new_file_content": "a = 2\n",
            },
            {
                "header": "{'file': {'path': 'pkg/tests/test_mod.py'}}",
                "old_file_content": "x\n",
                "new_file_content": "y\n",
            },
        ]
    }
    diff = gold_diff(commit)
    assert "+a = 2" in diff and "test_mod" not in diff
