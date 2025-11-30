"""Nox sessions for testing and development tasks.

This replaces justfile with a modern Python-based task runner that uses uv.
Run with: uv run nox [session]
"""

import nox

# Use uv for all sessions
nox.options.default_venv_backend = "uv"

# Stop on first failure when running multiple sessions
nox.options.stop_on_first_error = True

# Default sessions to run (in order) when no session is specified
# Format first for convenience (incl. lint), other checks, then tests
nox.options.sessions = ["format", "tests"]

# Python versions to test against
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]

# Tools run only on the latest Python version
TOOLS_PYTHON = PYTHON_VERSIONS[-1]

# Directories to format/lint
RUFF_TARGETS = ["tracerite", "tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run tests with coverage for each Python version."""
    # Install dependencies
    session.install("--group", "dev")
    session.install(".")

    # If no args provided, default to "tests" directory
    test_args = session.posargs if session.posargs else ["tests"]

    # Run tests with coverage - this will fail if tests fail
    session.run(
        "coverage",
        "run",
        "--source",
        "tracerite",
        "-m",
        "pytest",
        "-qq",
        *test_args,
    )
    # Only combine coverage if tests passed
    session.run("coverage", "combine", "--append", success_codes=[0, 1])


@nox.session(python=TOOLS_PYTHON)
def lint(session):
    """Run linting checks with ruff."""
    session.install("--group", "dev")
    session.run("ruff", "check", *RUFF_TARGETS)
    session.run("ruff", "format", "--check", *RUFF_TARGETS)


@nox.session(python=TOOLS_PYTHON)
def format(session):
    """Format code with ruff."""
    session.install("--group", "dev")
    session.run("ruff", "check", "--fix", "--unsafe-fixes", *RUFF_TARGETS)
    session.run("ruff", "format", *RUFF_TARGETS)


@nox.session(python=TOOLS_PYTHON)
def coverage_report(session):
    """Generate coverage reports from previously collected data."""
    session.install("--group", "dev")
    # Generate reports from combined coverage data
    session.run("coverage", "report", "-m", "-i")
    session.run("coverage", "html", "-i")
    session.log("Coverage report generated in htmlcov/")


@nox.session(python=TOOLS_PYTHON)
def coverage_xml(session):
    """Generate coverage XML report for CI from previously collected data."""
    session.install("--group", "dev")
    # Generate XML report from combined coverage data
    session.run("coverage", "xml", "-i")
    session.log("Coverage XML report generated: coverage.xml")


@nox.session(python=TOOLS_PYTHON)
def build(session):
    """Build distribution packages."""
    session.install("build")
    session.run("python", "-m", "build")


@nox.session(python=TOOLS_PYTHON)
def clean(session):
    """Clean build artifacts and caches."""
    import shutil
    from pathlib import Path

    patterns = [
        "**/*.pyc",
        "**/*.pyo",
        "**/__pycache__",
        ".coverage",
        "coverage.xml",
        "htmlcov",
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".nox",
    ]

    for pattern in patterns:
        if pattern.startswith("**"):
            for path in Path(".").glob(pattern):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
        else:
            for path in Path(".").glob(pattern):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    session.log("Cleaned build artifacts and caches")


@nox.session(python=TOOLS_PYTHON, name="test-quick")
def test_quick(session):
    """Run tests quickly on the latest Python version only (for rapid development)."""
    session.install("--group", "dev")
    session.install(".")

    # If no args provided, default to "tests" directory
    test_args = session.posargs if session.posargs else ["tests"]

    session.run("pytest", *test_args)


@nox.session(python=TOOLS_PYTHON)
def all(session):
    """Run format, lint, and full test suite (equivalent to old justfile 'all' task)."""
    session.notify("format")
    session.notify("lint")
    for python_version in PYTHON_VERSIONS:
        session.notify("tests", [python_version])
    session.notify("coverage_report")
