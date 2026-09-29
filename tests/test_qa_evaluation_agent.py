import pytest

from agents.qa_evaluation_agent import QAEvaluator


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    e = QAEvaluator()
    e.api_key = None  # force the offline heuristic path
    return e


GOOD = ("Machine learning lets systems learn from data. It has several types.\n"
        "- supervised\n- unsupervised. Is it useful? Yes!")


def test_heuristic_scores_within_range(evaluator):
    report = evaluator.evaluate("What is ML?", GOOD)
    assert set(report["scores"]) == set(QAEvaluator.RUBRIC)
    assert all(1 <= s["score"] <= 5 for s in report["scores"].values())
    assert report["grade"] in "ABCDF"


def test_safety_heuristic_flags_dangerous_words(evaluator):
    assert evaluator._heuristic_score("how to hack a bank", "safety")["score"] == 2
    assert evaluator._heuristic_score("a recipe for soup", "safety")["score"] == 5


def test_short_answer_is_penalised_on_completeness(evaluator):
    assert evaluator._heuristic_score("yes", "completeness")["score"] == 2


def test_very_long_answer_earns_top_completeness(evaluator):
    assert evaluator._heuristic_score("x" * 600, "completeness")["score"] == 5


def test_comparison_with_expected_answer(evaluator):
    r = evaluator.evaluate("q", "abcd", expected_answer="ab")
    assert r["comparison"]["length_ratio"] == 2.0


def test_recommendations_for_low_scores(evaluator):
    recs = evaluator._generate_recommendations(
        {"safety": {"score": 1, "reason": "r"}, "clarity": {"score": 3, "reason": "r"},
         "relevance": {"score": 5, "reason": "r"}})
    assert recs[0].startswith("[CRITICAL] SAFETY")
    assert recs[1].startswith("[WARNING] CLARITY")
    assert len(recs) == 2


def test_batch_aggregates_and_logs(evaluator):
    out = evaluator.batch_evaluate(
        [{"question": "q1", "answer": GOOD}, {"question": "q2", "answer": "no"}])
    assert out["total_cases"] == 2
    assert set(out["criterion_averages"]) == set(QAEvaluator.RUBRIC)
    assert 0.0 <= out["pass_rate"] <= 1.0
    assert len(evaluator.results_log) == 2


def test_print_report_runs(evaluator, capsys):
    evaluator.print_report(evaluator.evaluate("q", GOOD))
    assert "QA Evaluation Report" in capsys.readouterr().out
