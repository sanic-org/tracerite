"""Nox sessions for testing and development tasks.

This replaces justfile with a modern Python-based task runner that uses uv.
Run with: uv run nox [session]
"""

import shutil
from pathlib import Path

import nox

# Use uv for all sessions
nox.options.default_venv_backend = "uv"

# Stop on first failure when running multiple sessions
nox.options.stop_on_first_error = True

# Isolate and prevent running system-installed or external venv tooling.
nox.options.error_on_external_run = True

# Default sessions to run (in order) when no session is specified
# Format first for convenience (incl. lint), clean coverage, run tests, then report
nox.options.sessions = ["format", "clean_coverage", "test", "coverage_report"]

# Python versions to test against
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]

# Tools run only on the latest Python version
TOOLS_PYTHON = PYTHON_VERSIONS[-1]


@nox.session(python=PYTHON_VERSIONS)
def test(session):
    """Run tests with coverage for each Python version."""
    # Install dependencies
    session.install(".", "--group", "test")

    # If no args provided, default to "tests" directory
    test_args = session.posargs if session.posargs else ["tests"]

    # Run tests with coverage using parallel mode to avoid conflicts
    # Use --parallel-mode to write separate .coverage.* files
    session.run(
        "coverage",
        "run",
        "--parallel-mode",
        "--source",
        "tracerite",
        "-m",
        "pytest",
        "-qq",
        *test_args,
    )


@nox.session(python=TOOLS_PYTHON)
def lint(session):
    """Run linting checks with ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python=TOOLS_PYTHON)
def format(session):
    """Format code with ruff."""
    session.install("ruff")
    session.run("ruff", "check", "--fix", "--unsafe-fixes", ".")
    session.run("ruff", "format", ".")


@nox.session(python=TOOLS_PYTHON)
def clean_coverage(session):
    """Clean all coverage data and reports."""
    for path in Path(".").glob(".coverage*"):
        if path.is_file():
            path.unlink()

    htmlcov_path = Path("htmlcov")
    if htmlcov_path.exists():
        shutil.rmtree(htmlcov_path)

    coverage_xml = Path("coverage.xml")
    if coverage_xml.exists():
        coverage_xml.unlink()


@nox.session(python=TOOLS_PYTHON)
def coverage_report(session):
    """Generate coverage reports from previously collected data."""
    session.install("coverage")
    # Combine all parallel coverage data files, appending to existing .coverage file
    # This allows accumulating coverage from multiple test runs
    session.run("coverage", "combine", "--append")
    # Generate reports from combined coverage data
    session.run("coverage", "report", "-m")
    session.run("coverage", "html")


@nox.session(python=TOOLS_PYTHON)
def coverage_xml(session):
    """Generate coverage XML report for CI from previously collected data."""
    session.install("coverage")
    # Combine all parallel coverage data files if not already combined
    session.run("coverage", "combine", "--append", success_codes=[0, 1])
    # Generate XML report from combined coverage data
    session.run("coverage", "xml")


@nox.session(python=TOOLS_PYTHON)
def clean(session):
    """Clean build artifacts and caches."""
    # First clean coverage data and reports
    session.notify("clean_coverage")

    patterns = [
        "**/*.pyc",
        "**/*.pyo",
        "**/__pycache__",
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
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


@nox.session(python=TOOLS_PYTHON, name="test-latest")
def test_latest(session):
    """Run tests quickly on the latest Python version only (for rapid development)."""
    session.install(".", "--group", "test")

    # If no args provided, default to "tests" directory
    test_args = session.posargs if session.posargs else ["tests"]

    session.run("pytest", *test_args)
