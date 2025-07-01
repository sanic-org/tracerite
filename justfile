lint:
    uv run ruff check tracerite tests
    uv run ruff format --check tracerite tests
    #uv run mypy tracerite

format:
    uv run ruff format tracerite tests
    uv run ruff check --fix --unsafe-fixes tracerite tests

clean:
    rm -rf .coverage coverage.xml htmlcov

test-pyver ver:
    uv run --locked --isolated --python={{ver}} pytest -qq --cov=tracerite --cov-append --cov-report=

test: clean lint
    uv run just test-pyver 3.8
    uv run just test-pyver 3.9
    uv run just test-pyver 3.10
    uv run just test-pyver 3.11
    uv run just test-pyver 3.12
    uv run just test-pyver 3.13
    uv run coverage html

all: format test
