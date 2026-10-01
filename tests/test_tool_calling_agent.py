import sys
import types

import pytest
from langchain_core.messages import AIMessage

from agents import tool_calling_agent as tca


class FakeToolLLM:
    """Returns scripted AIMessages in order, recording what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def _patch_llm(monkeypatch, replies, tools=None):
    llm = FakeToolLLM(replies)
    monkeypatch.setattr(
        tca, "build_tool_agent", lambda model=None, tools_=None: (llm, tools or tca.TOOLS)
    )
    return llm


def test_calculator_basic_and_math_functions():
    assert tca.calculator.invoke({"expression": "15/100 * 2340"}) == "Result: 351.0"
    assert tca.calculator.invoke({"expression": "sqrt(144)"}) == "Result: 12.0"
    assert tca.calculator.invoke({"expression": "max(1, 5, 3)"}) == "Result: 5"


def test_calculator_reports_errors_instead_of_raising():
    out = tca.calculator.invoke({"expression": "1 / 0"})
    assert out.startswith("Error evaluating")
    assert "division by zero" in out


def test_calculator_has_no_builtins():
    out = tca.calculator.invoke({"expression": "open('/etc/passwd')"})
    assert out.startswith("Error evaluating")


def test_read_file_returns_contents(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert tca.read_file.invoke({"filepath": str(f)}) == "hello"


def test_read_file_missing_and_too_large(tmp_path):
    assert "File not found" in tca.read_file.invoke({"filepath": str(tmp_path / "nope")})
    big = tmp_path / "big.txt"
    big.write_text("x" * 50_001)
    assert "too large" in tca.read_file.invoke({"filepath": str(big)})


def test_current_datetime_format():
    from datetime import datetime

    out = tca.current_datetime.invoke({})
    datetime.strptime(out.split(" ")[0], "%Y-%m-%d")


def test_builtin_tool_registry():
    assert {t.name for t in tca.TOOLS} == {
        "calculator",
        "web_search",
        "read_file",
        "current_datetime",
    }


def test_loop_returns_directly_when_no_tool_calls(monkeypatch):
    _patch_llm(monkeypatch, [AIMessage(content="done")])
    assert tca.run_agent_loop("hi") == "done"


def test_loop_executes_tool_then_answers(monkeypatch):
    call = {"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}
    llm = _patch_llm(
        monkeypatch,
        [AIMessage(content="", tool_calls=[call]), AIMessage(content="It is 4")],
    )
    assert tca.run_agent_loop("2+2?") == "It is 4"
    tool_msg = llm.seen[1][-1]
    assert tool_msg.content == "Result: 4"
    assert tool_msg.tool_call_id == "c1"


def test_loop_handles_unknown_tool(monkeypatch):
    call = {"name": "rm_rf", "args": {}, "id": "c1"}
    llm = _patch_llm(
        monkeypatch,
        [AIMessage(content="", tool_calls=[call]), AIMessage(content="ok")],
    )
    tca.run_agent_loop("x")
    assert llm.seen[1][-1].content == "Unknown tool: rm_rf"


def test_loop_stops_at_max_iterations(monkeypatch):
    call = {"name": "current_datetime", "args": {}, "id": "c"}
    replies = [AIMessage(content="thinking", tool_calls=[call]) for _ in range(3)]
    llm = _patch_llm(monkeypatch, replies)
    out = tca.run_agent_loop("x", max_iterations=3)
    assert out.startswith("Max iterations reached")
    assert llm.replies == []  # exactly max_iterations model calls were made


def test_calculator_rejects_sandbox_escapes():
    for expr in [
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('os').system('true')",
        "[x for x in range(3)]",
        "'a' * 3",
        "9 ** 9 ** 9",
    ]:
        assert tca.calculator.invoke({"expression": expr}).startswith("Error evaluating"), expr


def test_calculator_supports_constants_and_unary():
    assert tca.calculator.invoke({"expression": "-pi"}) == f"Result: {-3.141592653589793}"
    assert tca.calculator.invoke({"expression": "2 ** 10"}) == "Result: 1024"


# ── web_search retry tests ─────────────────────────────────────────────────────


def _fake_tavily(monkeypatch, client_cls):
    monkeypatch.setitem(
        sys.modules, "tavily", types.SimpleNamespace(TavilyClient=client_cls)
    )


def test_retry_helper_raises_last_error_after_max_attempts():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        tca._call_with_retry(boom, max_attempts=2, base_delay=0)
    assert calls["n"] == 2


def test_web_search_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}

    class FlakyClient:
        def __init__(self, api_key=None):
            pass

        def search(self, query, max_results=3):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("transient failure")
            return {
                "results": [
                    {
                        "title": "Recovered",
                        "content": "x" * 400,
                        "url": "https://example.com",
                    }
                ]
            }

    _fake_tavily(monkeypatch, FlakyClient)
    out = tca.web_search.invoke({"query": "something"})
    assert "**Recovered**" in out
    assert attempts["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[0] < sleeps[1]  # backoff grows between attempts


def test_web_search_reports_error_after_retries_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class DeadClient:
        def __init__(self, api_key=None):
            pass

        def search(self, query, max_results=3):
            raise TimeoutError("service down")

    _fake_tavily(monkeypatch, DeadClient)
    out = tca.web_search.invoke({"query": "something"})
    assert out.startswith("Search error")
    assert "service down" in out


def test_web_search_missing_tavily_reports_install_hint(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name == "tavily":
            raise ImportError("no module")
        return _real_import(name, *args, **kwargs)

    _real_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    out = tca.web_search.invoke({"query": "something"})
    assert "Tavily is not installed" in out


# ── per-call token usage & cost logging ───────────────────────────────────────


def _usage_message(content="", tool_calls=None, in_tok=100, out_tok=20):
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata={
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    )


def test_loop_logs_usage_into_usage_log(monkeypatch, capsys):
    call = {"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}
    _patch_llm(
        monkeypatch,
        [
            _usage_message(tool_calls=[call], in_tok=100, out_tok=10),
            _usage_message(content="It is 4", in_tok=50, out_tok=5),
        ],
    )
    usage_log = []
    answer = tca.run_agent_loop("2+2?", usage_log=usage_log)

    assert answer == "It is 4"
    assert len(usage_log) == 2
    assert usage_log[0]["round"] == 1
    assert usage_log[0]["model"] == "gpt-4o-mini"
    assert usage_log[0]["prompt_tokens"] == 100
    assert usage_log[0]["completion_tokens"] == 10
    # 100 in + 10 out at gpt-4o-mini prices
    expected_first = 100 / 1000 * 0.00015 + 10 / 1000 * 0.0006
    assert usage_log[0]["cost_usd"] == pytest.approx(expected_first)
    assert usage_log[1]["round"] == 2

    out = capsys.readouterr().out
    assert "LLM call 1" in out and "LLM call 2" in out
    assert "Usage:" in out


def test_loop_usage_missing_metadata_does_not_crash(monkeypatch, capsys):
    # Plain AIMessages carry no usage metadata; the loop should still work.
    _patch_llm(monkeypatch, [AIMessage(content="done")])
    usage_log = []
    assert tca.run_agent_loop("hi", usage_log=usage_log) == "done"
    assert usage_log == [
        {
            "round": 1,
            "model": "gpt-4o-mini",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }
    ]


def test_loop_usage_uses_model_pricing(monkeypatch):
    _patch_llm(monkeypatch, [_usage_message(content="ok", in_tok=1000, out_tok=1000)])
    usage_log = []
    tca.run_agent_loop("hi", model="gpt-4o", usage_log=usage_log)
    expected = 1000 / 1000 * 0.0025 + 1000 / 1000 * 0.010
    assert usage_log[0]["cost_usd"] == pytest.approx(expected)
    assert usage_log[0]["model"] == "gpt-4o"
