import json

from benchmarks.r2egym.eval_infer import compute_reward, mismatched_tests


LOG = """
=========================== short test summary info ============================
PASSED test_a.py::test_ok
FAILED test_a.py::test_issue - AssertionError: wrong message
"""


def test_mismatched_tests_lists_status_differences():
    # R2E-Gym's parser drops the file prefix, so expected keys are bare names.
    expected = json.dumps({"test_ok": "PASSED", "test_issue": "PASSED"})
    assert compute_reward(LOG, expected) == 0.0
    assert mismatched_tests(LOG, expected) == {"test_issue": "FAILED/PASSED"}

    expected_more = json.dumps(
        {"test_ok": "PASSED", "test_issue": "FAILED", "test_new": "PASSED"}
    )
    assert mismatched_tests(LOG, expected_more) == {"test_new": "MISSING/PASSED"}
