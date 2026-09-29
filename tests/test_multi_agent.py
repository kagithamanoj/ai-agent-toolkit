from types import SimpleNamespace

from agents import multi_agent as ma


class FakeLLM:
    def __init__(self, review="APPROVED"):
        self.review = review

    def invoke(self, messages):
        text = messages[0].content
        if "senior editor" in text:
            return SimpleNamespace(content=self.review)
        if "research analyst" in text:
            return SimpleNamespace(content="NOTES")
        return SimpleNamespace(content="DRAFT")


def _state(**kw):
    base = {"topic": "t", "research": "", "draft": "", "review": "",
            "final_output": "", "iteration": 0}
    base.update(kw)
    return base


def test_should_continue_approved_finishes():
    assert ma.should_continue(_state(review="approved, nice", iteration=1)) == "finish"


def test_should_continue_revises_until_round_limit():
    assert ma.should_continue(_state(review="fix intro", iteration=1)) == "revise"
    assert ma.should_continue(_state(review="fix intro", iteration=2)) == "finish"


def test_nodes_fill_their_state_fields(monkeypatch):
    monkeypatch.setattr(ma, "get_llm", lambda *a, **k: FakeLLM())
    s = ma.researcher_agent(_state())
    s = ma.writer_agent(s)
    s = ma.reviewer_agent(s)
    assert (s["research"], s["draft"], s["review"], s["iteration"]) == (
        "NOTES", "DRAFT", "APPROVED", 1)


def test_graph_runs_end_to_end_when_approved(monkeypatch):
    monkeypatch.setattr(ma, "get_llm", lambda *a, **k: FakeLLM("APPROVED"))
    out = ma.build_multi_agent_graph().invoke(_state(topic="agents"))
    assert out["draft"] == "DRAFT" and out["iteration"] == 1


def test_graph_loop_is_bounded_when_never_approved(monkeypatch):
    monkeypatch.setattr(ma, "get_llm", lambda *a, **k: FakeLLM("needs work"))
    out = ma.build_multi_agent_graph().invoke(_state())
    assert out["iteration"] == 2  # stopped by the round limit, not by approval


def test_sequential_fallback(monkeypatch):
    monkeypatch.setattr(ma, "get_llm", lambda *a, **k: FakeLLM())
    out = ma.run_sequential("topic")
    assert out["final_output"] == "DRAFT" and out["iteration"] == 1
