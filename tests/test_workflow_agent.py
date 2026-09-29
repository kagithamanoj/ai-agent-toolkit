import pytest

from agents import workflow_agent as wa
from agents.workflow_agent import WorkflowAgent


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(wa.time, "sleep", lambda s: None)


def test_execution_order_follows_dependencies():
    wf = WorkflowAgent()
    wf.add_step("c", lambda **k: 3, depends_on=["b"])
    wf.add_step("b", lambda **k: 2, depends_on=["a"])
    wf.add_step("a", lambda: 1)
    assert wf._resolve_execution_order() == ["a", "b", "c"]


def test_cycle_is_detected():
    wf = WorkflowAgent()
    wf.add_step("a", lambda **k: 1, depends_on=["b"])
    wf.add_step("b", lambda **k: 1, depends_on=["a"])
    with pytest.raises(ValueError, match="Circular"):
        wf._resolve_execution_order()


def test_unknown_dependency_is_rejected():
    wf = WorkflowAgent()
    wf.add_step("a", lambda **k: 1, depends_on=["ghost"])
    with pytest.raises(ValueError, match="unknown step"):
        wf._resolve_execution_order()


def test_results_flow_between_steps_and_inputs():
    wf = WorkflowAgent("demo")
    wf.add_step("load", lambda url: url.upper(), inputs=["url"])
    wf.add_step("size", lambda load: len(load), depends_on=["load"])
    report = wf.run(url="abc")
    assert report["state"] == {"load": "ABC", "size": 3}
    assert report["completed"] == 2 and report["failed"] == 0


def test_transient_failure_is_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    wf = WorkflowAgent().add_step("f", flaky, max_retries=2)
    report = wf.run()
    assert calls["n"] == 3
    assert report["state"]["f"] == "ok" and report["failed"] == 0


def test_permanent_failure_is_reported_not_raised():
    def bad():
        raise ValueError("nope")

    wf = WorkflowAgent().add_step("bad", bad, max_retries=1)
    report = wf.run()
    assert report["failed"] == 1
    assert wf.steps["bad"].status == "failed"
    assert "nope" in report["execution_log"][0]["error"]


def test_visualize_lists_steps_with_status():
    wf = WorkflowAgent("v").add_step("a", lambda: 1).add_step("b", lambda **k: 2, depends_on=["a"])
    text = wf.visualize()
    assert "Step 1: a" in text and "(<- a)" in text and "[ ]" in text
    wf.run()
    assert "[x]" in wf.visualize()
