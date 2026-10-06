#!/usr/bin/env python3
"""Paired bootstrap significance test for SWE-bench Verified resolve rates.

Replicates Berg-Kirkpatrick, Burkett & Klein (EMNLP 2012), "An Empirical Investigation
of Statistical Significance in NLP", Sec. 2.2 / Fig. 1. If the winner beats the other
system by delta(x) > 0 on the n instances x, draw b samples x(i) of n instances from x
with replacement and estimate the one-sided p-value(x) as the share of x(i) with
delta(x(i)) >= 2 delta(x): the x(i) are centred on delta(x), not on H0's 0. b defaults
to the paper's 10^6; a pair without a gain gets p = 1.

That p is for the observed winner, a direction chosen after seeing the data. To claim
that two models differ, use the two-sided p, the share of x(i) with
|delta(x(i)) - delta(x)| >= |delta(x)|, which roughly agrees with the 95% percentile
interval of the same bootstrap gains. No multiple-comparison correction is applied.

Ties: Sec. 2.2's text counts gains "of delta(x) or greater", its formulas and Fig. 1
use ">". With 0/1 labels ties have real mass; they are counted by default (slightly
conservative) and excluded with `--strict` (slightly anti-conservative).

Each instance enters the gain only through d = resolved_A - resolved_B in {-1, 0, 1}
(its sufficient statistic, cf. footnote 4), so a bootstrap sample is exactly a
Multinomial(n, shares of -1/0/1) draw and gains compare as exact integer sums.

Labels come from `<model>/predictions.jsonl` in the results dataset, all read at one
revision; each model needs one graded row per SWE-bench Verified instance.

Usage:
  bootstrap_significance.py A B                      # one pair, gain = A - B
  bootstrap_significance.py --baseline B M1 M2 ...   # each model vs B
  bootstrap_significance.py --all-pairs M1 M2 M3     # every pair
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO = "synthetic-code-training/swebench-verified-results"
GOLDEN_PATCHES_PATH = Path(__file__).parent / "golden_patches.json"


def load_resolved(
    model: str, repo: str, revision: str, repo_files: set[str], instance_ids: list[str]
) -> np.ndarray:
    """0/1 resolved vector of `model` over `instance_ids`, checked against its eval_report.json."""
    if not {f"{model}/predictions.jsonl", f"{model}/eval_report.json"} <= repo_files:
        raise SystemExit(f"{model}: not graded in {repo}")
    path = hf_hub_download(
        repo, f"{model}/predictions.jsonl", repo_type="dataset", revision=revision
    )
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if sorted(r["instance_id"] for r in rows) != instance_ids:
        raise SystemExit(
            f"{model}: rows do not cover the {len(instance_ids)} instances once each"
        )
    if any(r.get("resolved") is None for r in rows):
        raise SystemExit(
            f"{model}: resolved labels missing on HF (re-uploaded after grading?)"
        )

    resolved = {r["instance_id"] for r in rows if r["resolved"]}
    report = hf_hub_download(
        repo, f"{model}/eval_report.json", repo_type="dataset", revision=revision
    )
    expected = json.loads(Path(report).read_text())["resolved_instances"]
    if len(resolved) != expected:
        raise SystemExit(
            f"{model}: {len(resolved)} resolved rows, eval_report.json says {expected}"
        )
    return np.array([i in resolved for i in instance_ids], dtype=np.int64)


def paired_bootstrap(
    d: np.ndarray, n_samples: int, seed: int, strict: bool
) -> dict[str, float]:
    """p-values and 95% percentile interval of the gain d.mean(), for d = A - B."""
    n, observed = len(d), int(d.sum())
    shares = np.bincount(d + 1, minlength=3) / n  # of -1, 0, 1
    counts = np.random.default_rng(seed).multinomial(n, shares, size=n_samples)
    sums = counts[:, 2] - counts[:, 0]  # n * delta(x(i)), exact integers
    lo, hi = np.percentile(sums, [2.5, 97.5]) / n
    if observed == 0:
        p_one = p_two = 1.0
    else:
        shifted = sums - observed  # centred on H0's 0
        bar = abs(observed) + strict  # on integers, "> k" is ">= k + 1"
        p_one = float((np.sign(observed) * shifted >= bar).mean())
        p_two = float((np.abs(shifted) >= bar).mean())
    return {
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def format_p(p: float, n_samples: int) -> str:
    return f"<{1 / n_samples:.0e}" if p == 0 else f"{p:.3g}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("models", nargs="+", metavar="MODEL")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline", metavar="MODEL", help="compare every model against it"
    )
    mode.add_argument("--all-pairs", action="store_true", help="compare every pair")
    p.add_argument("--samples", type=int, default=10**6, help="b (paper: 10^6)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strict", action="store_true", help="count only > 2 delta(x), as in Fig. 1"
    )
    p.add_argument("--repo", default=REPO)
    p.add_argument("--json", metavar="PATH", help="also write the results to this file")
    args = p.parse_args()

    if args.baseline:
        pairs = [(m, args.baseline) for m in args.models]
    elif args.all_pairs:
        pairs = list(itertools.combinations(args.models, 2))
    elif len(args.models) == 2:
        pairs = [(args.models[0], args.models[1])]
    else:
        raise SystemExit("give exactly two models, or --baseline, or --all-pairs")
    pairs = list(dict.fromkeys((a, b) for a, b in pairs if a != b))
    if not pairs:
        raise SystemExit("no pair of distinct models to compare")

    api = HfApi()
    revision = api.dataset_info(args.repo).sha or "main"  # one snapshot for every read
    repo_files = set(
        api.list_repo_files(args.repo, repo_type="dataset", revision=revision)
    )
    instance_ids = sorted(json.loads(GOLDEN_PATCHES_PATH.read_text()))
    labels = {
        m: load_resolved(m, args.repo, revision, repo_files, instance_ids)
        for m in dict.fromkeys(itertools.chain.from_iterable(pairs))
    }

    print(
        f"paired bootstrap, {args.repo}@{revision[:7]}: n={len(instance_ids)}, "
        f"b={args.samples:,}, seed={args.seed}, ties {'excluded' if args.strict else 'counted'}"
    )
    print("gain = A - B; p1: one-sided, for the observed winner; p2: two-sided")
    print(
        f"{'res A':>5} {'res B':>5} {'gain':>7} {'A only':>6} {'B only':>6} "
        f"{'p1':>7} {'p2':>7}  {'95% CI':<16}  A vs B"
    )
    results: list[dict[str, object]] = []
    for a, b in pairs:
        d = labels[a] - labels[b]
        # seeded per pair, so a pair's result does not depend on the other models listed
        stats = paired_bootstrap(d, args.samples, args.seed, args.strict)
        res_a, res_b = int(labels[a].sum()), int(labels[b].sum())
        a_only, b_only = int((d == 1).sum()), int((d == -1).sum())
        ci = f"[{stats['ci95_low'] * 100:+.1f}, {stats['ci95_high'] * 100:+.1f}]pp"
        print(
            f"{res_a:>5} {res_b:>5} {d.mean() * 100:>+5.1f}pp {a_only:>6} {b_only:>6} "
            f"{format_p(stats['p_one_sided'], args.samples):>7} "
            f"{format_p(stats['p_two_sided'], args.samples):>7}  {ci:<16}  {a} vs {b}"
        )
        results.append(
            {
                "model_a": a,
                "model_b": b,
                "resolved_a": res_a,
                "resolved_b": res_b,
                "gain": float(d.mean()),
                "a_only": a_only,
                "b_only": b_only,
                **stats,
            }
        )

    if args.json:
        out = {
            "repo": args.repo,
            "revision": revision,
            "n": len(instance_ids),
            "samples": args.samples,
            "seed": args.seed,
            "strict": args.strict,
            "pairs": results,
        }
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
