import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.hybridgym_verifyrepair.run_infer import apply_candidate, load_candidates


@dataclass
class _Result:
    exit_code: int
    stdout: str
    stderr: str


class _LocalWorkspace:
    """Runs commands with local bash, standing in for the container workspace."""

    def execute_command(self, command: str) -> _Result:
        res = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        return _Result(res.returncode, res.stdout, res.stderr)

    def file_upload(self, source_path: str, destination_path: str) -> None:
        shutil.copyfile(source_path, destination_path)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def f():\n    return 'a'\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


def test_candidate_applied_as_uncommitted_change(repo: Path):
    (repo / "mod.py").write_text('def f():\n    return "it\'s $HOME `x`"\n')
    patch = subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(["git", "checkout", "-q", "mod.py"], cwd=repo, check=True)

    apply_candidate(_LocalWorkspace(), f"{repo}/", patch)

    assert (repo / "mod.py").read_text() == 'def f():\n    return "it\'s $HOME `x`"\n'
    staged = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert staged == ""


def test_candidate_new_files_visible_in_git_diff(repo: Path):
    (repo / "new_mod.py").write_text("X = 1\n")
    subprocess.run(["git", "add", "-N", "new_mod.py"], cwd=repo, check=True)
    patch = subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(["git", "rm", "-q", "--cached", "new_mod.py"], cwd=repo, check=True)
    (repo / "new_mod.py").unlink()

    apply_candidate(_LocalWorkspace(), str(repo), patch)

    diff = subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert "new_mod.py" in diff


def test_non_applying_candidate_raises(repo: Path):
    bad = "--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n-nope\n+yes\n"
    with pytest.raises(RuntimeError, match="did not apply"):
        apply_candidate(_LocalWorkspace(), str(repo), bad)


def test_load_candidates_rejects_duplicates_and_empty(tmp_path: Path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"instance_id": "a", "candidate_patch": "p"}) + "\n")
    assert load_candidates(str(path)) == {"a": "p"}

    path.write_text(
        "\n".join(
            json.dumps({"instance_id": "a", "candidate_patch": p}) for p in ("p", "q")
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_candidates(str(path))

    path.write_text(json.dumps({"instance_id": "a", "candidate_patch": " "}))
    with pytest.raises(ValueError, match="Empty"):
        load_candidates(str(path))


def test_build_candidates_mix_and_repo_cap():
    from benchmarks.hybridgym_verifyrepair.build_candidates import build_candidates

    flawed = {f"numpy__{i}": f"bad{i}" for i in range(30)}
    flawed.update({f"aiohttp__{i}": f"bad{i}" for i in range(6)})
    resolved = {f"numpy__{i}": f"good{i}" for i in range(20, 60)}
    resolved.update({f"pandas__{i}": f"good{i}" for i in range(4)})
    sources = {"weak": (flawed, resolved)}

    rows = build_candidates(sources, correct_fraction=0.25, max_per_repo=None)
    correct = [r for r in rows if r["candidate_resolved"]]
    assert len(rows) - len(correct) == 36  # every flawed instance is used once
    assert len(correct) == 12  # 25% of the final mix
    assert all(r["instance_id"] not in flawed for r in correct)
    assert len({r["instance_id"] for r in rows}) == len(rows)
    assert rows == build_candidates(sources, 0.25, None)  # deterministic

    capped = build_candidates(sources, 0.25, max_per_repo=5)
    per_repo = {
        repo: sum(r["repo"] == repo for r in capped)
        for repo in ("numpy", "aiohttp", "pandas")
    }
    assert per_repo["numpy"] == 5 and per_repo["aiohttp"] == 5
    # 10 flawed -> 3 correct wanted; numpy is full, so they come from pandas.
    assert sum(bool(r["candidate_resolved"]) for r in capped) == 3
    assert per_repo["pandas"] == 3

    with pytest.raises(ValueError):
        build_candidates(sources, 1.0, None)
