"""Re-judge baseline trajectories with the LLM judge and overwrite verdicts/scores."""

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

BASELINE_DIR = TOOLS_DIR / "exp0" / "baseline"


def main() -> None:
    from nonfncall_judge import build_judge_rows, judge_rows, compute_scores
    from openai import OpenAI

    rows_path = BASELINE_DIR / "sampled_outputs.jsonl"
    verdicts_path = BASELINE_DIR / "verdicts.jsonl"
    scores_path = BASELINE_DIR / "scores.json"

    print(f"Loading sampled rows from {rows_path}...")
    sampled = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(sampled)} rows.")

    judge_rows_data = build_judge_rows(sampled)

    print(f"Judging {len(judge_rows_data)} trajectories with LLM ({JUDGE_MODEL})...")
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    verdicts = judge_rows(judge_rows_data, client, JUDGE_MODEL, workers=1, request_delay=2.0)

    verdicts_path.write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    print(f"Wrote {len(verdicts)} verdicts to {verdicts_path}")

    scores = compute_scores(verdicts)
    scores_path.write_text(json.dumps(scores, indent=2))
    print(f"Wrote scores to {scores_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
