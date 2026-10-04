"""
Agent YAML config file support.

A config file sets defaults for agent runs without repeating long CLI
flags. The file is plain YAML with up to three keys:

    model: gpt-4o-mini        # LLM model name
    max_rounds: 5             # max tool-calling rounds
    tools:                    # subset of built-in tools to enable
      - calculator
      - web_search
      - read_file
      - current_datetime

Any key may be omitted; missing keys fall back to the built-in
defaults. CLI flags still take precedence over values from the file.
"""

from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - guarded at call time
    yaml = None

# Built-in defaults: model, rounds, and the full set of built-in tools.
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_ROUNDS = 5
DEFAULT_TOOLS = ["calculator", "web_search", "read_file", "current_datetime"]

_KNOWN_KEYS = {"model", "max_rounds", "tools"}


def _require_yaml():
    if yaml is None:
        raise ImportError(
            "Reading a YAML config file needs PyYAML. "
            "Install it with: pip install pyyaml"
        )


def load_agent_config(path):
    """Load and validate an agent YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A dict with keys ``model`` (str), ``max_rounds`` (int) and
        ``tools`` (list of tool-name strings), all resolved against
        the built-in defaults.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a mapping, has unknown keys,
            or any value fails validation.
        ImportError: If PyYAML is not installed.
    """
    _require_yaml()

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path} must contain a YAML mapping, "
            f"got {type(data).__name__}"
        )

    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Config file {path} has unknown keys: {', '.join(unknown)}. "
            f"Supported keys: {', '.join(sorted(_KNOWN_KEYS))}"
        )

    model = data.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Config file {path}: 'model' must be a non-empty string")

    max_rounds = data.get("max_rounds", DEFAULT_MAX_ROUNDS)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise ValueError(f"Config file {path}: 'max_rounds' must be an integer")
    if max_rounds < 1:
        raise ValueError(f"Config file {path}: 'max_rounds' must be at least 1")

    tools = data.get("tools", DEFAULT_TOOLS)
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"Config file {path}: 'tools' must be a non-empty list")
    unknown_tools = [t for t in tools if t not in DEFAULT_TOOLS]
    if unknown_tools:
        raise ValueError(
            f"Config file {path}: unknown tools: {', '.join(unknown_tools)}. "
            f"Available tools: {', '.join(DEFAULT_TOOLS)}"
        )
    if len(set(tools)) != len(tools):
        raise ValueError(f"Config file {path}: 'tools' contains duplicates")

    return {"model": model, "max_rounds": max_rounds, "tools": list(tools)}
