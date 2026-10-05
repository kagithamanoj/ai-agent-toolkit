"""Tests for the offline evaluation harness (examples/eval_harness.py).

Everything must run with no API keys and no network: the harness uses
deterministic stub LLMs, a lexical fake embedding, and local fixture data.
"""

from examples import eval_harness as eh


def _snapshot(reports):
    return {
        name: (rep.passed, rep.total, dict(rep.metrics))
        for name, rep in reports.items()
    }


def test_run_all_covers_seven_patterns_offline():
    reports = eh.run_all()
    assert set(reports) == {
        "rag", "tool-calling", "multi-agent", "memory",
        "qa-eval", "web-scrape", "workflow",
    }
    for name, rep in reports.items():
        assert rep.total >= 1, name
        assert all(r.passed for r in rep.results), name


def test_results_are_deterministic():
    first = _snapshot(eh.run_all())
    second = _snapshot(eh.run_all())
    assert first == second


def test_pattern_filter_runs_only_selected():
    reports = eh.run_all("rag")
    assert list(reports) == ["rag"]
    assert reports["rag"].total == 4


def test_print_table_lists_all_patterns(capsys):
    reports = eh.run_all("memory")
    eh.print_table(reports)
    out = capsys.readouterr().out
    assert "EVAL HARNESS COMPARISON TABLE" in out
    assert "memory" in out
    assert "TOTAL" in out


def test_list_flag_shows_task_set(capsys):
    assert eh.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in ("rag", "tool-calling", "multi-agent", "memory",
                 "qa-eval", "web-scrape", "workflow"):
        assert name in out


def test_main_returns_zero_when_all_pass(capsys):
    assert eh.main([]) == 0


def test_memory_recall_proves_history_wiring():
    rep = eh.run_memory_tasks()
    recall = next(r for r in rep.results if r.task == "recall name across turns")
    assert recall.passed
    # The stub can only answer with the name if MemoryAgent actually passed
    # the earlier turn back in through the chain payload.
    assert "Ada" in recall.detail


def test_tool_calling_answers_come_from_real_tool_output():
    rep = eh.run_tool_calling_tasks()
    assert rep.metrics["tool calls executed"] == "3/3"
    assert all("calculator executed: yes" in r.detail for r in rep.results)


def test_rag_retrieval_surfaces_expected_facts():
    rep = eh.run_rag_tasks()
    assert rep.metrics["retrieval hit rate"] == "4/4"
    assert all("expected" in r.detail and "yes" in r.detail for r in rep.results)


def test_workflow_exercises_retry_and_dependencies():
    rep = eh.run_workflow_tasks()
    assert len(rep.results) == 1 and rep.results[0].passed
    assert rep.metrics["retries exercised"] == "1"
    assert "4 words; flaky=recovered" in rep.results[0].detail
