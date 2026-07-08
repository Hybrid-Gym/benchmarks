"""R2E-Gym dataset loading.

R2E-Gym datasets do not ship an ``instance_id`` column (they key on
``repo_name`` + ``commit_hash`` + ``docker_image``). The shared evaluation
machinery (``--select`` filtering, per-instance output dedup/resume, and the
agent-server image tag) all assumes a stable ``instance_id``, so this module
loads the split and synthesizes one before any filtering happens.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.utils.dataset import get_dataset as _get_base_dataset, prepare_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def make_instance_id(repo_name: str, commit_hash: str) -> str:
    """Synthesize a stable, unique instance id, e.g. ``aiohttp__f0d74880...``."""
    return f"{repo_name}__{commit_hash}"


def get_dataset(
    dataset_name: str,
    split: str,
    eval_limit: int | None = None,
    selected_instances_file: str | None = None,
) -> pd.DataFrame:
    """Load an R2E-Gym split with a synthesized ``instance_id`` column.

    Selection (`--select`) and limiting (`--n-limit`) are applied *after* the id
    is injected so that filtering by instance id works even though the raw
    dataset has no such column.
    """
    # Load the full split first (no select / no limit) so we can inject the id.
    df = _get_base_dataset(
        dataset_name=dataset_name,
        split=split,
        eval_limit=None,
        selected_instances_file=None,
    )

    if "instance_id" not in df.columns:
        df = df.copy()
        df["instance_id"] = [
            make_instance_id(str(r["repo_name"]), str(r["commit_hash"]))
            for _, r in df.iterrows()
        ]

    dup = int(df["instance_id"].duplicated().sum())
    if dup:
        logger.warning(
            "R2E-Gym: dropping %d duplicate synthesized instance_ids "
            "(repo_name+commit_hash collision).",
            dup,
        )
        df = df.drop_duplicates(subset="instance_id").reset_index(drop=True)

    # Now apply select + limit (prepare_dataset keys off instance_id).
    return prepare_dataset(df, eval_limit, selected_instances_file)
