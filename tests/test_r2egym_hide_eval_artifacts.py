import subprocess
from pathlib import Path

from benchmarks.r2egym.run_infer import hide_eval_artifacts_script


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_hides_tests_and_future_history(tmp_path: Path):
    repo = tmp_path / "testbed"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "future")
    (repo / "a.py").write_text("x = 2  # the fix\n")
    _git(repo, "commit", "-qam", "fix")
    fix = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "main")
    _git(repo, "tag", "v-future", fix)
    (repo / ".git" / "ORIG_HEAD").write_text(fix + "\n")
    hidden_tests = tmp_path / "r2e_tests"
    hidden_tests.mkdir()
    (repo / "run_tests.sh").write_text("pytest\n")

    script = hide_eval_artifacts_script(
        repo=str(repo), test_paths=(str(hidden_tests), str(repo / "run_tests.sh"))
    )
    subprocess.run(
        ["sh", "-c", script],
        check=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )

    assert not hidden_tests.exists() and not (repo / "run_tests.sh").exists()
    assert _git(repo, "log", "--all", "--format=%s").split() == ["base"]
    assert not (repo / ".git" / "ORIG_HEAD").exists()
    assert fix not in _git(repo, "fsck", "--unreachable", "--no-reflogs")
    assert subprocess.run(["git", "cat-file", "-e", fix], cwd=repo).returncode != 0
