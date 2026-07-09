"""Iterative prompt optimization pipeline for the funclocalize agent task.

Steps:
  0. Sample 50 trajectories from the baseline HF dataset and judge them.
  Then iteratively:
  1. Write/revise a Jinja2 prompt based on rubric scores.
  2. Run rollout with the new prompt (50 instances, gpt-5-mini).
  3. Convert rollout output to HF dataset format (combine_completions + convert_and_push).
  4. Judge the new trajectories.
  5. Check convergence (all rubrics >90% or error rate halved vs baseline).
  6. If not converged, go to step 1 (max 5 iterations).

Usage (from /home/yiqingxi/benchmarks):
  uv run python tools/funclocalize_prompt/pipeline.py [--exp-id EXP_ID] [--max-iter 5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── environment setup ─────────────────────────────────────────────────────────

BENCHMARKS_ROOT = Path(__file__).parent.parent.parent.resolve()
TOOLS_DIR = Path(__file__).parent

# Add benchmarks root to sys.path so we can import from tools/
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

# Constants
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
JUDGE_MODEL = "openai/openai/gpt-5-mini"
WRITER_MODEL = "openai/openai/gpt-5-mini"

BASELINE_DATASET = "synthetic-code-training/func_localize_gpt5mini_1346i"
ROLLOUT_DATASET = "hybrid-gym/hybrid_gym_func_localize"
LLM_CONFIG = str(BENCHMARKS_ROOT / ".llm_config" / "gpt5_mini.json")

RUNTIME_API_KEY = os.environ.get("RUNTIME_API_KEY") or os.environ.get("REMOTE_KEY", "")
RUNTIME_API_URL = os.environ.get("RUNTIME_API_URL", "https://runtime.eval.all-hands.dev")
EVAL_SERVER_IMAGE = os.environ.get(
    "OPENHANDS_EVAL_AGENT_SERVER_IMAGE", "ghcr.io/yiqingxyq/eval-agent-server"
)
IMAGE_TAG_PREFIX = os.environ.get("IMAGE_TAG_PREFIX", "e212d45")

STRATEGY_KEYS = ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing")


# ── helpers ────────────────────────────────────────────────────────────────────


def _env() -> dict:
    """Environment variables for subprocess calls."""
    e = dict(os.environ)
    e["RUNTIME_API_KEY"] = RUNTIME_API_KEY
    e["RUNTIME_API_URL"] = RUNTIME_API_URL
    e["OPENHANDS_EVAL_AGENT_SERVER_IMAGE"] = EVAL_SERVER_IMAGE
    e["IMAGE_TAG_PREFIX"] = IMAGE_TAG_PREFIX
    e["OPENHANDS_SUPPRESS_BANNER"] = "1"
    e["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    # Forward full image override if set (needed when default tag's manifest is broken)
    if os.environ.get("FULL_EVAL_AGENT_SERVER_IMAGE"):
        e["FULL_EVAL_AGENT_SERVER_IMAGE"] = os.environ["FULL_EVAL_AGENT_SERVER_IMAGE"]
    return e


def _save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2))


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pct(score: float) -> str:
    return f"{100 * score:.1f}%"


def _print_scores(scores: dict, label: str = "") -> None:
    hdr = f"=== {label} ===" if label else "=== Scores ==="
    print(hdr)
    total = scores.get("total", 0)
    for k in STRATEGY_KEYS:
        v = scores.get(k, 0)
        print(f"  {k:35s} {_pct(v):>7s}  ({int(v*total)}/{total})")
    print(f"  errors: {scores.get('errors', 0)}")


def _check_converged(baseline_scores: dict, iter_scores: dict) -> tuple[bool, str]:
    """Return (converged, reason).

    Converged if ALL rubrics are either:
      - > 90% compliance, OR
      - error rate reduced by ≥ 50% vs baseline
    """
    reasons = []
    all_good = True
    for k in STRATEGY_KEYS:
        b = baseline_scores.get(k, 0)
        c = iter_scores.get(k, 0)
        if c >= 0.90:
            reasons.append(f"{k}: {_pct(c)} (>90%)")
        elif (1 - c) <= 0.5 * (1 - b):  # error rate halved
            reasons.append(f"{k}: {_pct(c)} (error rate halved from {_pct(1-b)})")
        else:
            all_good = False
            reasons.append(f"{k}: {_pct(c)} (still below threshold)")
    return all_good, "; ".join(reasons)


# ── step 0: sample and judge baseline ─────────────────────────────────────────


def step0_sample_and_judge_baseline(
    out_dir: Path,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Load HF baseline dataset, sample n_samples, judge, save verdicts + scores."""
    from nonfncall_judge import build_judge_rows, judge_rows, compute_scores, aggregate, print_aggregate
    from openai import OpenAI

    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = out_dir / "verdicts.jsonl"
    scores_path = out_dir / "scores.json"
    rows_path = out_dir / "sampled_rows.jsonl"

    if scores_path.exists() and verdicts_path.exists():
        print(f"[Step 0] Cached baseline results found at {out_dir}, skipping.")
        return json.loads(scores_path.read_text())

    print(f"[Step 0] Loading baseline dataset: {BASELINE_DATASET}")
    from datasets import load_dataset

    ds = load_dataset(BASELINE_DATASET, split="train")
    print(f"[Step 0] Dataset has {len(ds)} rows. Sampling {n_samples}...")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    sampled = [ds[i] for i in indices]

    # Save sampled rows (without full messages to save space — just ids)
    _save_jsonl(rows_path, [{"instance_id": r["instance_id"], "resolved": r.get("resolved")} for r in sampled])

    # Convert to history-event format for judge.py
    judge_rows_data = build_judge_rows(sampled)

    print(f"[Step 0] Judging {len(judge_rows_data)} baseline trajectories (sequential)...")
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    verdicts = judge_rows(judge_rows_data, client, JUDGE_MODEL, workers=1, request_delay=2.0)

    _save_jsonl(verdicts_path, verdicts)

    scores = compute_scores(verdicts)
    _save_json(scores_path, scores)

    _print_scores(scores, "Baseline scores")
    return scores


# ── step 1: write / revise prompt ─────────────────────────────────────────────


def step1_write_prompt(
    out_dir: Path,
    iteration: int,
    baseline_scores: dict,
    prev_iter_scores: list[dict],
    baseline_verdicts: list[dict],
    baseline_trajectories: list[dict],
    iter_verdicts: list[dict],
    iter_trajectories: list[dict],
) -> Path:
    """Write (or revise) the prompt and save to out_dir/prompt.j2."""
    from prompt_writer import write_initial_prompt, revise_prompt

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompt.j2"

    if prompt_path.exists():
        print(f"[Step 1] Prompt already exists at {prompt_path}, skipping.")
        return prompt_path

    start_t = time.time()
    if iteration == 1:
        print("[Step 1] Writing initial prompt based on rubrics...")
        template = write_initial_prompt(api_key=NVIDIA_API_KEY, model=WRITER_MODEL)
    else:
        print(f"[Step 1] Revising prompt for iteration {iteration}...")
        prev_template = (out_dir.parent / f"iter_{iteration - 1}" / "prompt.j2").read_text()
        template = revise_prompt(
            api_key=NVIDIA_API_KEY,
            current_template=prev_template,
            baseline_scores=baseline_scores,
            prev_iter_scores=prev_iter_scores,
            baseline_verdicts=baseline_verdicts,
            baseline_trajectories=baseline_trajectories,
            iter_verdicts=iter_verdicts,
            iter_trajectories=iter_trajectories,
            iteration=iteration,
            model=WRITER_MODEL,
        )

    elapsed = time.time() - start_t
    print(f"[Step 1] Prompt written in {elapsed:.1f}s")

    prompt_path.write_text(template)
    print(f"[Step 1] Saved prompt to {prompt_path}")
    return prompt_path


# ── step 2: run rollout ────────────────────────────────────────────────────────


def step2_run_rollout(
    prompt_path: Path,
    out_dir: Path,
    n_limit: int = 50,
    note: str = "",
) -> Path:
    """Run hybridgym-funclocalize-infer with the given prompt. Returns output dir."""
    rollout_dir = out_dir / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = _find_output_jsonl(rollout_dir)
    if output_jsonl and output_jsonl.exists() and output_jsonl.stat().st_size > 0:
        # Verify it has valid data (not all errors)
        rows = _load_jsonl(output_jsonl)
        valid = [r for r in rows if not r.get("error")]
        if len(valid) >= n_limit:
            print(f"[Step 2] Rollout already done: {len(valid)} valid rows. Skipping.")
            return rollout_dir
        elif valid:
            print(f"[Step 2] Existing rollout has {len(valid)}/{n_limit} valid rows. Continuing.")
            # Don't skip — fall through to re-run (evaluation framework will skip already-done instances)
        else:
            print(f"[Step 2] Existing rollout has {len(rows)} rows but all errored. Re-running.")
            # Don't skip — fall through to re-run

    # Per-instance retry budget: 1 retry × startup_timeout s ≈ startup_timeout*2 total
    startup_timeout = int(os.environ.get("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600"))
    # Total budget: n_limit / 2 workers × 2 attempts × startup_timeout + agent time
    total_timeout = n_limit // 2 * 2 * startup_timeout + 3600

    num_workers = int(os.environ.get("EVAL_NUM_WORKERS", "1"))
    cmd = [
        "uv", "run", "hybridgym-funclocalize-infer",
        LLM_CONFIG,
        "--workspace", os.environ.get("EVAL_WORKSPACE", "remote"),
        "--num-workers", str(num_workers),
        "--max-iterations", "30",
        "--max-retries", "1",  # fail fast when cluster is unavailable
        "--n-limit", str(n_limit),
        "--prompt-path", str(prompt_path.resolve()),
        "--output-dir", str(rollout_dir.resolve()),
        "--note", note or f"prompt_opt",
    ]

    print(f"[Step 2] Running rollout with {n_limit} instances...")
    print(f"[Step 2] Command: {' '.join(cmd)}")
    print(f"[Step 2] Output dir: {rollout_dir}")

    result = subprocess.run(
        cmd,
        cwd=str(BENCHMARKS_ROOT),
        env=_env(),
        timeout=total_timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rollout failed with exit code {result.returncode}")

    output_jsonl = _find_output_jsonl(rollout_dir)
    if not output_jsonl:
        raise RuntimeError(f"No output.jsonl found in {rollout_dir}")

    print(f"[Step 2] Rollout complete. Output: {output_jsonl}")
    return rollout_dir


def _find_output_jsonl(rollout_dir: Path) -> Path | None:
    """Find the best output.jsonl in the structured eval output subdirectory.

    Prefers non-empty output.jsonl; falls back to output.critic_attempt_1.jsonl
    when output.jsonl is empty (critic mode: all instances wrote to critic file).
    """
    # Structured output: rollout_dir/<dataset>-<split>/<model>_<sdk>_maxiter_N/output.jsonl
    matches = list(rollout_dir.rglob("output.jsonl"))
    if not matches:
        return None

    # Prefer non-empty output.jsonl
    non_empty = [p for p in matches if p.stat().st_size > 0 and "critic" not in p.name]
    if non_empty:
        return non_empty[0]

    # Fall back to output.critic_attempt_1.jsonl in the same dirs
    for m in matches:
        critic = m.parent / "output.critic_attempt_1.jsonl"
        if critic.exists() and critic.stat().st_size > 0:
            return critic

    # Last resort: any non-empty file
    all_nonempty = [p for p in matches if p.stat().st_size > 0]
    return all_nonempty[0] if all_nonempty else matches[0]


# ── step 3: convert data ───────────────────────────────────────────────────────


def step3_convert_data(rollout_dir: Path, out_dir: Path) -> Path:
    """Run combine_completions and convert_and_push (dry-run) on the rollout output."""
    output_jsonl = _find_output_jsonl(rollout_dir)
    if not output_jsonl:
        raise RuntimeError(f"No output.jsonl in {rollout_dir}")

    # Step 3a: combine_completions.py
    with_completions = output_jsonl.with_suffix("").with_suffix(".with_completions.jsonl.gz")
    if not with_completions.exists():
        print(f"[Step 3a] Running combine_completions on {output_jsonl.name}...")
        script = BENCHMARKS_ROOT / "benchmarks" / "utils" / "post_process_scripts" / "combine_completions.py"
        result = subprocess.run(
            ["uv", "run", "python", str(script), str(output_jsonl)],
            cwd=str(BENCHMARKS_ROOT),
            env=_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[Step 3a] WARNING: combine_completions failed: {result.stderr[:500]}")
        else:
            print(f"[Step 3a] Done: {with_completions}")
    else:
        print(f"[Step 3a] Already exists: {with_completions}")

    # Step 3b: convert_and_push.py --dry-run
    local_dataset_jsonl = out_dir / "dataset.jsonl"
    if not local_dataset_jsonl.exists() and with_completions.exists():
        print(f"[Step 3b] Converting to HF format (dry-run, local save)...")
        script = BENCHMARKS_ROOT / "benchmarks" / "utils" / "post_process_scripts" / "convert_and_push.py"
        result = subprocess.run(
            [
                "uv", "run", "python", str(script),
                "--src", str(with_completions),
                "--repo", "synthetic-code-training/funclocalize_prompt_opt_local",
                "--out-jsonl", str(local_dataset_jsonl),
                "--dry-run",
            ],
            cwd=str(BENCHMARKS_ROOT),
            env=_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[Step 3b] WARNING: convert_and_push failed: {result.stderr[:500]}")
        else:
            print(f"[Step 3b] Local dataset saved: {local_dataset_jsonl}")
    else:
        if local_dataset_jsonl.exists():
            print(f"[Step 3b] Already exists: {local_dataset_jsonl}")
        else:
            print(f"[Step 3b] Skipping (no with_completions file)")

    return output_jsonl


# ── step 4: judge trajectories ─────────────────────────────────────────────────


def step4_judge_trajectories(output_jsonl: Path, out_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Judge trajectories from output.jsonl and return (scores, verdicts, judge_rows)."""
    from nonfncall_judge import compute_scores
    from openai import OpenAI

    verdicts_path = out_dir / "verdicts.jsonl"
    scores_path = out_dir / "scores.json"

    if scores_path.exists() and verdicts_path.exists():
        print(f"[Step 4] Cached verdicts found at {verdicts_path}, skipping.")
        verdicts = _load_jsonl(verdicts_path)
        scores = json.loads(scores_path.read_text())
        rows = _load_jsonl(output_jsonl)
        return scores, verdicts, rows

    print(f"[Step 4] Loading trajectories from {output_jsonl}...")
    # Load from output.jsonl (already in history-event format)
    raw_rows = _load_jsonl(output_jsonl)
    rows = [r for r in raw_rows if not r.get("error")]
    print(f"[Step 4] Loaded {len(rows)} valid trajectories (out of {len(raw_rows)})")

    print(f"[Step 4] Judging {len(rows)} trajectories...")
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

    from nonfncall_judge import judge_rows as _judge_rows_fn

    # Use workers=1 and request_delay=2.0; auto-fallback to rule-based on rate limits
    verdicts = _judge_rows_fn(rows, client, JUDGE_MODEL, workers=1, request_delay=2.0)

    _save_jsonl(verdicts_path, verdicts)
    scores = compute_scores(verdicts)
    _save_json(scores_path, scores)

    _print_scores(scores, "Iteration scores")
    return scores, verdicts, rows


# ── step 5: compute rollout cost ──────────────────────────────────────────────


def _compute_rollout_cost(output_jsonl: Path) -> dict:
    """Extract LLM cost from output.jsonl metrics."""
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    count = 0

    for line in output_jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        metrics = row.get("metrics") or {}
        total_cost += float(metrics.get("accumulated_cost", 0) or 0)
        usage = metrics.get("accumulated_usage") or {}
        total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        count += 1

    return {
        "count": count,
        "total_cost_usd": round(total_cost, 6),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(exp_id: str, max_iter: int = 5, n_samples: int = 50) -> None:
    exp_dir = TOOLS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"# Experiment: {exp_id}")
    print(f"# Max iterations: {max_iter}, Samples per run: {n_samples}")
    print(f"# Experiment dir: {exp_dir}")
    print(f"{'#'*60}\n")

    # ── Step 0: baseline ──────────────────────────────────────────────────────
    baseline_dir = exp_dir / "baseline"
    baseline_scores = step0_sample_and_judge_baseline(baseline_dir, n_samples=n_samples)
    _print_scores(baseline_scores, "Baseline")

    # Load baseline verdicts and trajectories for use in prompt writer
    baseline_verdicts = _load_jsonl(baseline_dir / "verdicts.jsonl") if (baseline_dir / "verdicts.jsonl").exists() else []
    # Build fake "trajectory" rows from HF dataset for failure-example extraction
    # (just a stub — we don't save full messages; use empty history so judge writer skips gracefully)
    baseline_trajectories: list[dict] = []
    if (baseline_dir / "sampled_rows.jsonl").exists():
        for r in _load_jsonl(baseline_dir / "sampled_rows.jsonl"):
            baseline_trajectories.append({"instance_id": r["instance_id"], "history": []})

    summary: dict = {
        "exp_id": exp_id,
        "baseline_scores": baseline_scores,
        "iterations": [],
    }

    prev_iter_scores: list[dict] = []
    prev_iter_verdicts: list[dict] = []
    prev_iter_trajectories: list[dict] = []

    # ── Iterative loop ────────────────────────────────────────────────────────
    for iteration in range(1, max_iter + 1):
        iter_label = f"iter_{iteration}"
        iter_dir = exp_dir / iter_label
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_meta: dict = {"iteration": iteration}

        print(f"\n{'='*60}")
        print(f"= ITERATION {iteration}")
        print(f"{'='*60}\n")

        # Step 1: Write/revise prompt
        prompt_path = step1_write_prompt(
            out_dir=iter_dir,
            iteration=iteration,
            baseline_scores=baseline_scores,
            prev_iter_scores=prev_iter_scores,
            baseline_verdicts=baseline_verdicts,
            baseline_trajectories=baseline_trajectories,
            iter_verdicts=prev_iter_verdicts,
            iter_trajectories=prev_iter_trajectories,
        )
        iter_meta["prompt_path"] = str(prompt_path)

        # Step 2: Run rollout
        rollout_dir = step2_run_rollout(
            prompt_path=prompt_path,
            out_dir=iter_dir,
            n_limit=n_samples,
            note=f"prompt_opt_{exp_id}_iter{iteration}",
        )
        iter_meta["rollout_dir"] = str(rollout_dir)

        # Step 3: Convert data
        output_jsonl = step3_convert_data(rollout_dir, iter_dir)
        iter_meta["output_jsonl"] = str(output_jsonl)

        # Compute rollout cost
        rollout_cost = _compute_rollout_cost(output_jsonl)
        iter_meta["rollout_cost"] = rollout_cost
        print(f"[Cost] Rollout: ${rollout_cost['total_cost_usd']:.4f} ({rollout_cost['count']} instances)")

        # Step 4: Judge trajectories
        scores, verdicts, trajectories = step4_judge_trajectories(output_jsonl, iter_dir)
        iter_meta["scores"] = scores
        _print_scores(scores, f"Iteration {iteration}")

        prev_iter_scores.append(scores)
        prev_iter_verdicts = verdicts
        prev_iter_trajectories = trajectories

        # Step 5: Check convergence
        converged, reason = _check_converged(baseline_scores, scores)
        iter_meta["converged"] = converged
        iter_meta["convergence_reason"] = reason
        print(f"\n[Convergence] {'CONVERGED' if converged else 'NOT CONVERGED'}: {reason}")

        summary["iterations"].append(iter_meta)
        _save_json(exp_dir / "summary.json", summary)

        if converged:
            print(f"\n✓ Pipeline converged at iteration {iteration}!")
            break

        if iteration == max_iter:
            print(f"\n✗ Reached max iterations ({max_iter}) without convergence.")

    # Save final summary
    summary["final_scores"] = prev_iter_scores[-1] if prev_iter_scores else baseline_scores
    _save_json(exp_dir / "summary.json", summary)
    print(f"\nFinal summary saved to {exp_dir / 'summary.json'}")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment ID (default: exp_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum number of iterations (default: 5)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of trajectories per run (default: 50)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from an existing experiment directory",
    )
    args = parser.parse_args()

    if args.resume:
        exp_id = args.resume
    elif args.exp_id:
        exp_id = args.exp_id
    else:
        exp_id = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up sys.path for nonfncall_judge and prompt_writer
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))

    run_pipeline(exp_id=exp_id, max_iter=args.max_iter, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
