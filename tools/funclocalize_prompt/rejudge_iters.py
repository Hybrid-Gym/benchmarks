"""Re-judge iter_1 and iter_2 with LLM judge, overwrite verdicts/scores, and print avg steps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
BENCHMARKS_ROOT = TOOLS_DIR.parent.parent
sys.path.insert(0, str(TOOLS_DIR))

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
JUDGE_MODEL = "openai/openai/gpt-5-mini"

EXP_DIR = TOOLS_DIR / "exp0"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def find_output_jsonl(rollout_dir: Path) -> Path | None:
    matches = list(rollout_dir.rglob("output.jsonl"))
    if not matches:
        return None
    non_empty = [p for p in matches if p.stat().st_size > 0 and "critic" not in p.name]
    if non_empty:
        return non_empty[0]
    for m in matches:
        critic = m.parent / "output.critic_attempt_1.jsonl"
        if critic.exists() and critic.stat().st_size > 0:
            return critic
    all_nonempty = [p for p in matches if p.stat().st_size > 0]
    return all_nonempty[0] if all_nonempty else matches[0]


def avg_steps(rows: list[dict]) -> float:
    """Average number of ActionEvents per trajectory (excludes errored rows)."""
    valid = [r for r in rows if not r.get("error")]
    if not valid:
        return 0.0
    counts = [sum(1 for ev in (r.get("history") or []) if ev.get("kind") == "ActionEvent") for r in valid]
    return sum(counts) / len(counts)


def avg_steps_baseline(sampled: list[dict]) -> float:
    """Avg steps for baseline (HF message format — count assistant turns with tool calls)."""
    from nonfncall_judge import nonfncall_messages_to_history
    counts = []
    for row in sampled:
        history = nonfncall_messages_to_history(row.get("messages") or [])
        counts.append(sum(1 for ev in history if ev.get("kind") == "ActionEvent"))
    return sum(counts) / len(counts) if counts else 0.0


def rejudge_iter(iter_name: str, client) -> dict:
    from nonfncall_judge import judge_rows, compute_scores

    iter_dir = EXP_DIR / iter_name
    rollout_dir = iter_dir / "rollout"
    verdicts_path = iter_dir / "verdicts.jsonl"
    scores_path = iter_dir / "scores.json"

    output_jsonl = find_output_jsonl(rollout_dir)
    if not output_jsonl:
        raise RuntimeError(f"No output.jsonl found for {iter_name}")

    print(f"\n[{iter_name}] Loading from {output_jsonl.name}...")
    raw = load_jsonl(output_jsonl)
    rows = [r for r in raw if not r.get("error")]
    print(f"[{iter_name}] {len(rows)} valid rows (of {len(raw)} total)")

    print(f"[{iter_name}] Judging with LLM...")
    verdicts = judge_rows(rows, client, JUDGE_MODEL, workers=1, request_delay=2.0)

    verdicts_path.write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    scores = compute_scores(verdicts)
    scores_path.write_text(json.dumps(scores, indent=2))

    steps = avg_steps(rows)
    print(f"[{iter_name}] Scores: {json.dumps(scores)}")
    print(f"[{iter_name}] Avg steps: {steps:.1f}")
    return scores, steps


def main() -> None:
    from nonfncall_judge import build_judge_rows
    from openai import OpenAI

    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

    # Baseline avg steps (already judged)
    baseline_sampled = load_jsonl(EXP_DIR / "baseline" / "sampled_outputs.jsonl")
    baseline_steps = avg_steps_baseline(baseline_sampled)
    baseline_scores = json.loads((EXP_DIR / "baseline" / "scores.json").read_text())

    # Re-judge iter_1 and iter_2
    iter1_scores, iter1_steps = rejudge_iter("iter_1", client)
    iter2_scores, iter2_steps = rejudge_iter("iter_2", client)

    # iter_3 already has LLM verdicts — just load scores and compute steps
    iter3_output = find_output_jsonl(EXP_DIR / "iter_3" / "rollout")
    iter3_rows = [r for r in load_jsonl(iter3_output) if not r.get("error")]
    iter3_steps = avg_steps(iter3_rows)
    iter3_scores = json.loads((EXP_DIR / "iter_3" / "scores.json").read_text())

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'':20s} {'broad_then_narrow':>18} {'multi_round':>12} {'read_after':>11} {'avg_steps':>10}")
    print("-" * 70)
    for label, scores, steps in [
        ("baseline", baseline_scores, baseline_steps),
        ("iter_1",   iter1_scores,   iter1_steps),
        ("iter_2",   iter2_scores,   iter2_steps),
        ("iter_3",   iter3_scores,   iter3_steps),
    ]:
        print(
            f"{label:20s}"
            f" {scores['broad_then_narrow']*100:>17.1f}%"
            f" {scores['multi_round_refinement']*100:>11.1f}%"
            f" {scores['read_after_narrowing']*100:>10.1f}%"
            f" {steps:>10.1f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
