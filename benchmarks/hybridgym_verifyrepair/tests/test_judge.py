from typing import Any

import pytest

from benchmarks.hybridgym_verifyrepair.build_candidates import build_candidates
from benchmarks.hybridgym_verifyrepair.judge_eval import (
    final_answer,
    parse_verdict,
    score,
)
from benchmarks.hybridgym_verifyrepair.run_infer import request_verdict


@pytest.mark.parametrize(
    "message, expected",
    [
        ("The fix works.\nVERDICT: CORRECT", True),
        ("It misses the empty case.\nVERDICT: INCORRECT\n", False),
        ("**VERDICT:** INCORRECT", False),
        ("verdict: correct", True),
        ("VERDICT: INCORRECT\n...on second thought\nVERDICT: CORRECT", True),
        ("The changes look correct.", None),
        ("VERDICT: NOT CORRECT", None),
        ("", None),
    ],
)
def test_parse_verdict(message: str, expected: bool | None):
    assert parse_verdict(message) is expected


CANDIDATE = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-a\n+b\n"


def _row(iid: str, message: str | None, patch: str = CANDIDATE) -> dict[str, Any]:
    history: list[dict[str, Any]] = [
        {
            "kind": "ActionEvent",
            "tool_name": "terminal",
            "action": {"command": "cd /testbed && python reproduce_issue.py"},
        }
    ]
    if message is not None:
        history.append(
            {
                "kind": "ActionEvent",
                "tool_name": "finish",
                "action": {"message": message},
            }
        )
    return {"instance_id": iid, "history": history, "test_result": {"git_patch": patch}}


def test_score_judge_rollouts():
    candidates = {
        iid: {"candidate_patch": CANDIDATE, "candidate_resolved": label}
        for iid, label in [("a", True), ("b", False), ("c", False), ("d", True)]
    }
    leftover = CANDIDATE + "diff --git a/r.py b/r.py\n+++ b/r.py\n@@ -0,0 +1 @@\n+x\n"
    rows = [
        _row("a", "VERDICT: INCORRECT"),
        _row("a", "VERDICT: CORRECT"),  # retry of a; the last row wins
        _row("b", "VERDICT: INCORRECT", patch=leftover),
        _row("c", "VERDICT: CORRECT"),
        _row("d", None),  # never finished
    ]

    report = score(rows, candidates)

    assert report["resolved_ids"] == ["a", "b"]
    assert report["accuracy"] == 0.5
    assert report["confusion"] == {
        "correct/no_verdict": 1,
        "correct/said_correct": 1,
        "flawed/said_correct": 1,
        "flawed/said_incorrect": 1,
    }
    by_id = {r["instance_id"]: r for r in report["results"]}
    assert by_id["a"]["patch_unchanged"] and not by_id["b"]["patch_unchanged"]
    assert all(r["ran_code"] for r in report["results"])


def test_build_candidates_paired():
    flawed = {f"numpy__{i}": f"bad{i}" for i in range(40)}
    resolved = {f"numpy__{i}": f"good{i}" for i in range(20, 60)}
    sources = {"weak": (flawed, resolved)}

    rows = build_candidates(sources, 0.5, None, paired=True)

    assert {r["instance_id"] for r in rows} == {f"numpy__{i}" for i in range(20, 40)}
    assert sum(bool(r["candidate_resolved"]) for r in rows) == 10
    for r in rows:
        kind = "good" if r["candidate_resolved"] else "bad"
        assert r["candidate_patch"] == kind + str(r["instance_id"]).split("__")[1]
    assert rows == build_candidates(sources, 0.5, None, paired=True)


def _message(source: str, text: str) -> dict[str, Any]:
    return {
        "kind": "MessageEvent",
        "source": source,
        "llm_message": {"content": [{"type": "text", "text": text}]},
    }


def test_final_answer_takes_the_last_finish_or_agent_message():
    finish = _row("x", "VERDICT: INCORRECT")["history"]
    assert final_answer(finish) == "VERDICT: INCORRECT"
    # Answered with a plain message instead of calling finish.
    assert final_answer(finish + [_message("agent", "VERDICT: CORRECT")]) == (
        "VERDICT: CORRECT"
    )
    # The last action is a tool call that is not finish: no answer yet.
    assert final_answer(_row("x", None)["history"]) == ""


class _Event:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def model_dump(self, mode: str) -> dict[str, Any]:
        return self.data


class _FakeConversation:
    """Replies to each reminder with the next canned agent message."""

    def __init__(self, first_answer: str, replies: list[str]):
        self.state = self
        self.events = [_Event(_message("agent", first_answer))]
        self.replies = replies
        self.sent: list[str] = []

    def send_message(self, text: str) -> None:
        self.sent.append(text)
        self.events.append(_Event(_message("user", text)))

    def run(self) -> None:
        self.events.append(_Event(_message("agent", self.replies.pop(0))))


@pytest.mark.parametrize(
    "first, replies, n_sent",
    [
        ("Looks right.\nVERDICT: CORRECT", [], 0),
        ("Looks right.", ["VERDICT: CORRECT"], 1),
        ("Looks right.", ["Done.", "Still done."], 2),  # gives up after two
    ],
)
def test_request_verdict(first: str, replies: list[str], n_sent: int):
    conversation = _FakeConversation(first, replies)
    request_verdict(conversation)  # pyright: ignore[reportArgumentType]
    assert len(conversation.sent) == n_sent
