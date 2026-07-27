"""
SWE-Gym benchmark configuration.

SWE-Gym rows use the SWE-bench instance schema, so defaults mirror the SWE-bench
config retargeted at the SWE-Gym/SWE-Gym train split.
"""

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "SWE-Gym/SWE-Gym",
    "split": "train",
    "num_workers": 30,
    **CONDENSER_DEFAULTS,
}
