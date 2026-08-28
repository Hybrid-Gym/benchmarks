#!/usr/bin/env python3
"""
Compare pairwise instance-id overlapping rate across multiple HuggingFace datasets.
Overlapping rate = num_overlap / total_instance_num (size of union)
"""

from datasets import load_dataset
from itertools import combinations

DATASETS = [
    "synthetic-code-training/func_localize_claude45_1457i",
    "synthetic-code-training/func_localize_claude47_1467i",
    "synthetic-code-training/func_localize_qwen35_35b_1348i",
    "synthetic-code-training/func_localize_qwen35_397b_1299i",
    "synthetic-code-training/func_localize_gpt5mini_1346i",
    "synthetic-code-training/func_localize_gpt55_1477i",
    "synthetic-code-training/func_localize_kimi_k25_1431i",
    "synthetic-code-training/func_localize_deepseek_v4_flash_1374i",
]

SHORT_NAMES = {
    "synthetic-code-training/func_localize_claude45_1457i": "claude45",
    "synthetic-code-training/func_localize_claude47_1467i": "claude47",
    "synthetic-code-training/func_localize_qwen35_35b_1348i": "qwen35_35b",
    "synthetic-code-training/func_localize_qwen35_397b_1299i": "qwen35_397b",
    "synthetic-code-training/func_localize_gpt5mini_1346i": "gpt5mini",
    "synthetic-code-training/func_localize_gpt55_1477i": "gpt55",
    "synthetic-code-training/func_localize_kimi_k25_1431i": "kimi_k25",
    "synthetic-code-training/func_localize_deepseek_v4_flash_1374i": "deepseek_v4_flash",
}


def load_instance_ids(dataset_name):
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    # Try common instance id column names
    for col in ["instance_id", "id", "task_id", "problem_id"]:
        if col in ds.column_names:
            ids = set(ds[col])
            print(f"  -> {len(ids)} unique instance IDs (column: '{col}')")
            return ids
    print(f"  Columns: {ds.column_names}")
    raise ValueError(f"No instance ID column found in {dataset_name}")


def main():
    # Load all datasets
    id_sets = {}
    for ds_name in DATASETS:
        short = SHORT_NAMES[ds_name]
        id_sets[short] = load_instance_ids(ds_name)

    names = list(id_sets.keys())
    n = len(names)

    print("\n" + "=" * 80)
    print("Dataset sizes (and size / 1500):")
    for name in names:
        size = len(id_sets[name])
        print(f"  {name}: {size}  ({size / 1500:.3f})")

    print("\n" + "=" * 80)
    print("Pairwise overlapping rates (overlap / union):\n")

    # Header
    col_w = 14
    header = f"{'':14}" + "".join(f"{name:>{col_w}}" for name in names)
    print(header)
    print("-" * len(header))

    # Matrix
    matrix = {}
    for a in names:
        row = f"{a:<14}"
        for b in names:
            if a == b:
                row += f"{'1.000':>{col_w}}"
                matrix[(a, b)] = 1.0
            else:
                overlap = len(id_sets[a] & id_sets[b])
                union = len(id_sets[a] | id_sets[b])
                rate = overlap / union if union > 0 else 0.0
                matrix[(a, b)] = rate
                row += f"{rate:>{col_w}.3f}"
        print(row)

    print("\n" + "=" * 80)
    print("Pairwise overlap details (sorted by overlap rate desc):\n")
    print(f"{'Dataset A':<20} {'Dataset B':<20} {'Overlap':>8} {'Union':>8} {'Rate':>8}")
    print("-" * 70)

    pairs = []
    for a, b in combinations(names, 2):
        overlap = len(id_sets[a] & id_sets[b])
        union = len(id_sets[a] | id_sets[b])
        rate = overlap / union if union > 0 else 0.0
        pairs.append((rate, overlap, union, a, b))

    for rate, overlap, union, a, b in sorted(pairs, reverse=True):
        print(f"{a:<20} {b:<20} {overlap:>8} {union:>8} {rate:>8.3f}")


if __name__ == "__main__":
    main()
