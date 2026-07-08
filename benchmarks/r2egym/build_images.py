#!/usr/bin/env python3
"""
Build agent-server images for all unique R2E-Gym base images in a dataset split.

Unlike SWE-Bench, R2E-Gym ships the base docker image in the dataset itself (the
``docker_image`` column, e.g. ``namanjain12/aiohttp_final:<commit>``), so the set
of base images is read straight from the dataset rather than derived from instance
ids. The three-phase build pipeline (shared builder image -> per-instance base
images -> assembled agent images) is reused verbatim from the SWE-Bench helpers.

Note: for local ``--workspace docker`` runs you do NOT need to pre-build with this
script; ``run_infer.py`` builds each agent-server image on demand. Pre-building is
only required for the ``remote`` / ``apptainer`` workspaces (which pull pre-pushed,
publicly accessible images from a registry).

Example:
  uv run python -m benchmarks.r2egym.build_images \
    --dataset R2E-Gym/R2E-Gym-Lite --split train \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import argparse
import sys
from typing import Any

from benchmarks.r2egym.dataset import get_dataset
from benchmarks.utils.build_utils import default_build_output_dir
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(row: Any) -> str:
    """Return the R2E-Gym base image for a dataset row (dict or pandas Series).

    R2E-Gym stores the fully-qualified image reference in the ``docker_image``
    column, e.g. ``namanjain12/aiohttp_final:f0d74880...``. It is used verbatim.
    """
    image = str(row["docker_image"]).strip()
    logger.debug(f"R2E-Gym base image: {image}")
    return image


def extract_custom_tag(base_image: str) -> str:
    """Derive a docker-tag-safe, per-instance-unique tag from a base image ref.

    Example:
        namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198
        -> aiohttp_final_f0d74880deec8fcd982bce639c93c5e130d41198

    The repository *and* the tag are both kept (joined with ``_``) so that two
    instances of the same repo at different commits never collide on the same
    assembled agent-server tag. Any characters outside ``[A-Za-z0-9_.-]`` are
    replaced with ``_`` to stay within docker tag constraints.
    """
    name_tag = base_image.split("/")[-1]  # drop registry/namespace
    name_tag = name_tag.replace(":", "_")
    safe = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in name_tag)
    return safe


def collect_unique_base_images(
    dataset,
    split,
    n_limit,
    selected_instances_file: str | None = None,
):
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted({get_official_docker_image(row) for _, row in df.iterrows()})


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build R2E-Gym agent-server images using the phased pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="R2E-Gym/R2E-Gym-Lite",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--image",
        default=EVAL_AGENT_SERVER_IMAGE,
        help="Target repo/name for final agent images",
    )
    parser.add_argument(
        "--target",
        default="source-minimal",
        help="Final image target tag suffix",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push built images to the registry",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Concurrent builds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per image build",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of images (0 = no limit)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Path to text file containing instance IDs to select",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild final images even if matching remote tags already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Reuse the shared phased-build helpers from the SWE-Bench module; they are
    # generic and accept a custom_tag_fn.
    from benchmarks.swebench.build_base_images import (
        assemble_all_agent_images,
        build_all_base_images,
        build_builder_image,
    )

    parser = get_parser()
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    builder_result = build_builder_image(push=args.push, force_build=args.force_build)
    if builder_result.error or not builder_result.tags:
        print(
            builder_result.error or "Builder image build produced no tags",
            file=sys.stderr,
        )
        return 1

    rc = build_all_base_images(
        base_images=base_images,
        build_dir=build_dir,
        push=args.push,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        force_build=args.force_build,
        custom_tag_fn=extract_custom_tag,
    )
    if rc != 0:
        return rc

    def custom_tag_fn(base: str) -> str:
        return extract_custom_tag(base)

    return assemble_all_agent_images(
        base_images=base_images,
        builder_tag=builder_result.tags[0],
        build_dir=build_dir,
        target_image=args.image,
        target=args.target,
        push=args.push,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        force_build=args.force_build,
        custom_tag_fn=custom_tag_fn,
    )


if __name__ == "__main__":
    sys.exit(main())
