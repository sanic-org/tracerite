"""Exception-message formatting scenarios for the console demo."""

from __future__ import annotations


def mixed_content() -> None:
    """Mixed message: short lines, long wrapped lines, inline code, and many lines."""
    msg = (
        "Configuration validation failed for the requested pipeline.\n"
        "\n"
        "The supplied manifest references several deprecated fields and "
        "contains a few sections that cannot be parsed automatically, so "
        "you will need to review them manually before the deployment can "
        "continue safely.\n"
        "\n"
        "Offending values:\n"
        "- `metadata.labels['app.kubernetes.io/very-long-component-name']` "
        "exceeds the maximum allowed length of 63 characters\n"
        "- `spec.template.spec.containers[0].resources.limits.cpu` is set to "
        "`1000000000000000000000000000000000000000000000000000000m` which is not a valid quantity\n"
        "- `spec.template.spec.containers[0].image` uses tag `latest`\n"
        "- `spec.replicas` is `0` which disables the service entirely\n"
        "\n"
        "Suggested fix:\n"
        "```python\n"
        "config = load_manifest('deployment.yaml')\n"
        "config['metadata']['labels']['app.kubernetes.io/component'] = 'api'\n"
        "config['spec']['replicas'] = max(1, config['spec']['replicas'])\n"
        "validate_and_apply(config)\n"
        "```\n"
        "\n"
        "For additional context, the full set of validation errors encountered "
        "while scanning the manifest is listed below. Each error includes the "
        "field path, the offending value, and a short explanation of why the "
        "value was rejected by the schema validator.\n"
        "\n"
        + "\n".join(
            f"[{i:03d}] validation error in field `spec.paths.{i}.method`: method "
            f"name is too long and contains invalid characters"
            for i in range(80)
        )
    )
    raise ValueError(msg)


def chained_multiline() -> None:
    """Chained exceptions where both have multi-line messages."""
    try:
        raise ValueError("Original problem\nwith extra detail")
    except ValueError as e:
        raise RuntimeError(
            "While handling the original error\na second failure occurred"
        ) from e

