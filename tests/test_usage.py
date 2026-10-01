"""Tests for utils/usage.py: pricing, usage extraction, aggregation."""

import pytest
from langchain_core.messages import AIMessage

from utils.usage import (
    cost_usd,
    record_call,
    summarize_calls,
    usage_from_message,
)


def test_cost_usd_known_model():
    # gpt-4o-mini: 0.00015 in / 0.0006 out per 1K
    assert cost_usd("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert cost_usd("gpt-4o-mini", 2000, 0) == pytest.approx(0.0003)


def test_cost_usd_unknown_model_uses_default():
    # default: 0.002 in / 0.008 out per 1K
    assert cost_usd("some-future-model", 1000, 1000) == pytest.approx(0.010)


def test_cost_usd_rejects_negative_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        cost_usd("gpt-4o-mini", -1, 0)


def test_usage_from_message_reads_metadata():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    assert usage_from_message(msg) == (120, 30)


def test_usage_from_message_without_metadata_returns_zeros():
    assert usage_from_message(AIMessage(content="hi")) == (0, 0)
    assert usage_from_message(object()) == (0, 0)


def test_record_call_and_summarize():
    calls = []
    msg = AIMessage(
        content="",
        usage_metadata={"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000},
    )
    record = record_call(calls, model="gpt-4o-mini", round_no=1, response=msg)
    assert record["round"] == 1
    assert record["model"] == "gpt-4o-mini"
    assert record["prompt_tokens"] == 1000
    assert record["completion_tokens"] == 1000
    assert record["cost_usd"] == pytest.approx(0.00075)
    assert len(calls) == 1

    summary = summarize_calls(calls)
    assert summary["calls"] == 1
    assert summary["total_tokens"] == 2000
    assert summary["total_cost_usd"] == pytest.approx(0.00075)


def test_summarize_empty_calls():
    assert summarize_calls([]) == {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0}
