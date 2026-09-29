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
