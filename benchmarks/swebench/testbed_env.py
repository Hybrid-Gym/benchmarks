"""Point the agent's shells at the SWE-bench ``testbed`` env and its working copy.

SWE-bench images activate the ``testbed`` conda env only in ``/root/.bashrc``, but
the agent-server runs as ``openhands``, whose PATH resolves ``python`` to the base
conda interpreter (no repository dependencies). Agents that try to run code or
tests therefore hit ``ModuleNotFoundError`` and fall back to syntax checks.

The harness also copies ``/testbed`` to ``/workspace/<repo>`` and the agent edits
the copy, while the env's editable install still points at ``/testbed``. Even with
the env active, test runners such as Django's ``tests/runtests.py`` would import the
unedited original, so the editable-install pointers are rewritten to the copy.
Repositories installed non-editably (e.g. requests) get ``PYTHONPATH`` instead.

The command is a no-op on images without a ``testbed`` env (e.g. SWE-bench Pro).
Validated on the SWE-bench Verified repositories (egg-link, easy-install.pth,
``__editable__`` pth and PEP 660 finder installs). Not handled: meson-python editable
installs (e.g. pandas >= 2.1, whose loader module has no plain "/testbed" pointer).
"""

TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"
TESTBED_ACTIVATE = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
)


def build_testbed_env_command(
    repo_path: str, testbed_python: str = TESTBED_PYTHON
) -> str:
    """Return a shell command that sets up the testbed env for the agent user.

    Args:
        repo_path: The agent's working copy of the repository (``/workspace/<repo>``).
        testbed_python: The testbed interpreter (overridable for tests).
    """
    repo = repo_path.rstrip("/")
    # "/testbed" followed by a non-path character or end of line, as it appears in
    # egg-link, easy-install.pth, __editable__ pth and PEP 660 finder MAPPING values.
    pointer = "/testbed([^A-Za-z0-9_.-]|$)"
    return " && ".join(
        [
            f"{{ [ -x {testbed_python} ] || exit 0; }}",
            f"sp=$({testbed_python} -c 'import site; print(site.getsitepackages()[0])')",
            f'files=$(grep -lE \'{pointer}\' "$sp"/*.pth "$sp"/*.egg-link '
            f'"$sp"/__editable__*finder.py 2>/dev/null || true)',
            f"if [ -n \"$files\" ]; then sudo -n sed -i -E 's#{pointer}#{repo}\\1#g' $files; "
            f"else echo 'export PYTHONPATH={repo}${{PYTHONPATH:+:$PYTHONPATH}}' >> ~/.bashrc; fi",
            f"echo '{TESTBED_ACTIVATE}' >> ~/.bashrc",
        ]
    )
