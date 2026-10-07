"""Tests for utils/agent_config.py and the --config CLI wiring."""

import sys

import pytest

from utils.agent_config import load_agent_config
from agents import tool_calling_agent as tca


def _write(tmp_path, text):
    f = tmp_path / "agent.yaml"
    f.write_text(text)
    return str(f)


def test_defaults_when_file_is_empty(tmp_path):
    cfg = load_agent_config(_write(tmp_path, ""))
    assert cfg == {
        "model": "gpt-4o-mini",
        "max_rounds": 5,
        "tools": ["calculator", "web_search", "read_file", "current_datetime"],
    }


def test_loads_full_config(tmp_path):
    cfg = load_agent_config(
        _write(
            tmp_path,
            "model: gpt-4o\n"
            "max_rounds: 3\n"
            "tools:\n"
            "  - calculator\n"
            "  - current_datetime\n",
        )
    )
    assert cfg["model"] == "gpt-4o"
    assert cfg["max_rounds"] == 3
    assert cfg["tools"] == ["calculator", "current_datetime"]


def test_partial_config_keeps_defaults(tmp_path):
    cfg = load_agent_config(_write(tmp_path, "model: gpt-4o\n"))
    assert cfg["model"] == "gpt-4o"
    assert cfg["max_rounds"] == 5


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_agent_config(str(tmp_path / "nope.yaml"))


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_agent_config(_write(tmp_path, "mode: gpt-4o\n"))


def test_not_a_mapping_raises(tmp_path):
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_agent_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_unknown_tool_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown tools: web_browse"):
        load_agent_config(_write(tmp_path, "tools:\n  - calculator\n  - web_browse\n"))


def test_bad_max_rounds_rejected(tmp_path):
    for bad in ("max_rounds: 0\n", "max_rounds: -2\n", "max_rounds: many\n"):
        with pytest.raises(ValueError, match="max_rounds"):
            load_agent_config(_write(tmp_path, bad))


def test_empty_tools_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty list"):
        load_agent_config(_write(tmp_path, "tools: []\n"))


def test_duplicate_tools_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicates"):
        load_agent_config(
            _write(tmp_path, "tools:\n  - calculator\n  - calculator\n")
        )


def test_blank_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="'model'"):
        load_agent_config(_write(tmp_path, "model: '   '\n"))


def test_resolve_tools_maps_names():
    tools = tca.resolve_tools(["calculator", "current_datetime"])
    assert [t.name for t in tools] == ["calculator", "current_datetime"]


def test_resolve_tools_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        tca.resolve_tools(["calculator", "nope"])


def _patch_run(monkeypatch):
    seen = {}

    def fake_run(query, model=None, max_iterations=None, tools=None, stream=False):
        seen["query"] = query
        seen["model"] = model
        seen["max_iterations"] = max_iterations
        seen["tools"] = None if tools is None else [t.name for t in tools]
        seen["stream"] = stream
        return "done"

    monkeypatch.setattr(tca, "run_agent_loop", fake_run)
    return seen


def test_cli_config_used_for_defaults(tmp_path, monkeypatch, capsys):
    cfg = _write(
        tmp_path, "model: gpt-4o\nmax_rounds: 2\ntools:\n  - calculator\n"
    )
    seen = _patch_run(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["tool_calling_agent", "-q", "hi", "-c", cfg])
    tca.main()
    assert seen == {
        "query": "hi",
        "model": "gpt-4o",
        "max_iterations": 2,
        "tools": ["calculator"],
        "stream": False,
    }


def test_cli_flags_override_config(tmp_path, monkeypatch, capsys):
    cfg = _write(tmp_path, "model: gpt-4o\nmax_rounds: 2\n")
    seen = _patch_run(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tool_calling_agent", "-q", "hi", "-c", cfg, "--model", "o3-mini",
         "--max-rounds", "7"],
    )
    tca.main()
    assert seen["model"] == "o3-mini"
    assert seen["max_iterations"] == 7
    # No tools key in config: falls back to all built-in tools.
    assert seen["tools"] == ["calculator", "web_search", "read_file", "current_datetime"]


def test_cli_without_config_uses_builtin_defaults(monkeypatch, capsys):
    seen = _patch_run(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["tool_calling_agent", "-q", "hi"])
    tca.main()
    assert seen["model"] == "gpt-4o-mini"
    assert seen["max_iterations"] == 5
    assert seen["tools"] is None


def test_cli_max_rounds_zero_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["tool_calling_agent", "-q", "hi", "--max-rounds", "0"]
    )
    with pytest.raises(SystemExit):
        tca.main()


def test_example_config_file_is_valid():
    import pathlib

    example = pathlib.Path(__file__).parent.parent / "examples" / "agent_config.yaml"
    cfg = load_agent_config(str(example))
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["max_rounds"] == 5
    assert len(cfg["tools"]) == 4
