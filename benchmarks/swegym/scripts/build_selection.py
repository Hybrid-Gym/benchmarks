"""Draw the shared SWE-Gym instance selection that every compared model runs.

Comparing models only means something if they attempt the SAME instances, so the
selection is built once, written to a file, and passed to each rollout via --select.

Two properties matter and neither is free:

  * **Reproducible** -- a fixed seed over the sorted id list, so re-running this
    reproduces the identical set rather than silently drifting the comparison.
  * **Runnable** -- restricted to instances whose xingyaoww image exists on Docker
    Hub. 37 of the 2438 train instances have none; leaving them in would make the
    supervisor's completion target unreachable for every model at once, and each
    rollout would burn its retry budget on the same dead instances.

Run probe_images.py first to populate the availability cache.

Usage:
    python benchmarks/swegym/scripts/build_selection.py [--n 1500] [--seed 42]
"""

import argparse
import collections
import json
import random
from typing import TYPE_CHECKING, cast

from datasets import load_dataset


if TYPE_CHECKING:
    from datasets import Dataset


CACHE = "eval_outputs/swegym_image_availability.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_outputs/swegym_select_1500.txt")
    ap.add_argument("--dataset", default="SWE-Gym/SWE-Gym")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    with open(CACHE, encoding="utf-8") as f:
        avail = json.load(f)
    unknown = [i for i, v in avail.items() if v is None]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} instances still unprobed; re-run probe_images.py "
            "(it resumes from the cache) before drawing a selection"
        )

    runnable = sorted(i for i, ok in avail.items() if ok is True)
    print(
        f"runnable: {len(runnable)} of {len(avail)} "
        f"(no image: {sum(1 for v in avail.values() if v is False)})"
    )
    if len(runnable) < args.n:
        raise SystemExit(f"only {len(runnable)} runnable instances, need {args.n}")

    sel = sorted(random.Random(args.seed).sample(runnable, args.n))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(i + "\n" for i in sel))
    print(f"wrote {len(sel)} ids (seed={args.seed}) -> {args.out}")

    ds = cast("Dataset", load_dataset(args.dataset, split=args.split))
    repo_of = dict(
        zip(cast("list[str]", ds["instance_id"]), cast("list[str]", ds["repo"]))
    )
    picked = collections.Counter(repo_of[i] for i in sel)
    pool = collections.Counter(repo_of[i] for i in runnable)
    print(f"\n{'repo':<32}{'pool':>7}{'sel':>7}{'share':>9}")
    for repo, n in pool.most_common():
        print(f"{repo:<32}{n:>7}{picked[repo]:>7}{100 * picked[repo] / n:>8.1f}%")


if __name__ == "__main__":
    main()
