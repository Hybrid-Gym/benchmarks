import subprocess
from pathlib import Path

import pytest

from benchmarks.swebench.testbed_env import TESTBED_ACTIVATE, build_testbed_env_command


@pytest.fixture
def fake_env(tmp_path: Path) -> dict[str, Path]:
    """A fake testbed: site-packages dir, an interpreter that reports it, a sudo shim."""
    site = tmp_path / "site-packages"
    site.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f"#!/bin/sh\necho {site}\n")
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n')
    for f in (python, sudo):
        f.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    return {"site": site, "python": python, "bin": bin_dir, "home": home}


def _run(cmd: str, env: dict[str, Path]) -> None:
    subprocess.run(
        ["bash", "-c", cmd],
        check=True,
        env={"PATH": f"{env['bin']}:/usr/bin:/bin", "HOME": str(env["home"])},
    )


def test_rewrites_editable_pointers_to_working_copy(fake_env):
    site = fake_env["site"]
    (site / "pkg.egg-link").write_text("/testbed\n.")
    (site / "easy-install.pth").write_text("/opt/other\n/testbed\n")
    (site / "__editable__.pytest-7.2.pth").write_text("/testbed/src\n")
    (site / "__editable___astropy_finder.py").write_text(
        "MAPPING = {'astropy': '/testbed/astropy'}\n"
    )
    (site / "unrelated.pth").write_text("/testbed2/keep\n")

    _run(
        build_testbed_env_command("/workspace/pkg/", str(fake_env["python"])), fake_env
    )

    assert (site / "pkg.egg-link").read_text() == "/workspace/pkg\n."
    assert (site / "easy-install.pth").read_text() == "/opt/other\n/workspace/pkg\n"
    assert (site / "__editable__.pytest-7.2.pth").read_text() == "/workspace/pkg/src\n"
    assert (
        "'/workspace/pkg/astropy'"
        in (site / "__editable___astropy_finder.py").read_text()
    )
    assert (site / "unrelated.pth").read_text() == "/testbed2/keep\n"
    bashrc = (fake_env["home"] / ".bashrc").read_text()
    assert bashrc.strip().splitlines()[-1] == TESTBED_ACTIVATE
    assert "PYTHONPATH" not in bashrc


def test_non_editable_install_falls_back_to_pythonpath(fake_env):
    _run(
        build_testbed_env_command("/workspace/requests", str(fake_env["python"])),
        fake_env,
    )

    lines = (fake_env["home"] / ".bashrc").read_text().strip().splitlines()
    assert lines[0].startswith("export PYTHONPATH=/workspace/requests")
    assert lines[-1] == TESTBED_ACTIVATE


def test_noop_without_testbed_env(fake_env, tmp_path):
    missing = tmp_path / "no-such-python"
    _run(build_testbed_env_command("/workspace/app", str(missing)), fake_env)

    assert not (fake_env["home"] / ".bashrc").exists()


def test_quote_terminated_pointer_is_rewritten(fake_env):
    site = fake_env["site"]
    finder = site / "__editable___pkg_finder.py"
    finder.write_text("MAPPING = {'pkg': '/testbed'}\nOTHER = '/testbed_other'\n")

    _run(build_testbed_env_command("/workspace/pkg", str(fake_env["python"])), fake_env)

    text = finder.read_text()
    assert "'pkg': '/workspace/pkg'" in text
    assert "'/testbed_other'" in text
