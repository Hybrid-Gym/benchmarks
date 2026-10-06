"""
R2E-Gym benchmark configuration.

Default values mirror the SWE-Bench benchmark config, retargeted at the R2E-Gym
datasets (R2E-Gym/R2E-Gym-Lite, R2E-Gym-Subset, R2E-Gym-V1).
"""

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 240,  # Maximum number of events before condensing
    "condenser_keep_first": 2,  # Number of initial events to always keep
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "R2E-Gym/R2E-Gym-Lite",
    "split": "train",
    "num_workers": 30,
    **CONDENSER_DEFAULTS,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "R2E-Gym/R2E-Gym-Lite",
    "split": "train",
    # Peak disk during eval is roughly workers x base-image size (each image is
    # removed as soon as its instance finishes), so keep this modest.
    "workers": 4,
    # In-image test-run timeout; 300s matches R2E-Gym's reward-calc default.
    "timeout": 300,
}
