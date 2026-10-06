"""
R2E-Gym hyperparameters and constant values.

This module provides constant values used in the R2E-Gym evaluation workflow.
For dataset, model, and worker defaults, see config.py (INFER_DEFAULTS, EVAL_DEFAULTS).
"""

from typing import Final, Literal


# Docker
# R2E-Gym ships the base image in the dataset (``docker_image`` column), so there
# is no image name derived from the instance id. The repo is at ``/testbed``.
REPO_PATH_IN_IMAGE: Final[str] = "/testbed"

# Build target type (matches openhands.agent_server.docker.build.TargetType)
TargetType = Literal["binary", "binary-minimal", "source", "source-minimal"]
BUILD_TARGET_SOURCE_MINIMAL: Final[TargetType] = "source-minimal"
BUILD_TARGET_BINARY: Final[TargetType] = "binary"
DEFAULT_BUILD_TARGET: Final[TargetType] = BUILD_TARGET_SOURCE_MINIMAL

# Runtime
DEFAULT_RUNTIME_API_URL: Final[str] = "https://runtime.eval.all-hands.dev"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT: Final[int] = 600


# Git
GIT_USER_EMAIL: Final[str] = "evaluation@openhands.dev"
GIT_USER_NAME: Final[str] = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE: Final[str] = "patch"
# Commit of the shipped /testbed state (setup changes + untracked harness files)
# taken before the agent runs, so the final diff excludes them.
GIT_BASE_SNAPSHOT_MESSAGE: Final[str] = "r2egym base snapshot"

# Patch Processing
# R2E-Gym patches are applied verbatim during evaluation; nothing is stripped.
SETUP_FILES_TO_REMOVE: Final[tuple[str, ...]] = ()
