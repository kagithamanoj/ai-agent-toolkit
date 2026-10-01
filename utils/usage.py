"""
Token usage and cost tracking for agent runs.

Prices follow the same idea as the llm-usage-tracker project: a small
built-in table of per-1K-token prices in USD, with a conservative
default for unknown models. Prices were last reviewed 2026-10-01;
override the table if they drift.
"""

from __future__ import annotations

# (input price, output price) per 1K tokens in USD.
PRICING_PER_1K_USD = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1": (0.002, 0.008),
}

DEFAULT_PRICE_PER_1K_USD = (0.002, 0.008)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of one LLM call from token counts."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts cannot be negative")
    in_price, out_price = PRICING_PER_1K_USD.get(model, DEFAULT_PRICE_PER_1K_USD)
    return prompt_tokens / 1000 * in_price + completion_tokens / 1000 * out_price


def usage_from_message(response) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) out of a LangChain AIMessage.

    Returns (0, 0) when the message carries no usage metadata, so fakes
    and offline providers never break the loop.
    """
    meta = getattr(response, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))


def record_call(calls: list, *, model: str, round_no: int, response) -> dict:
    """Append one per-call usage record to calls and return it."""
    prompt_tokens, completion_tokens = usage_from_message(response)
    record = {
        "round": round_no,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost_usd(model, prompt_tokens, completion_tokens), 6),
    }
    calls.append(record)
    return record


def summarize_calls(calls: list) -> dict:
    """Aggregate per-call records into a totals summary."""
    return {
        "calls": len(calls),
        "total_tokens": sum(
            c["prompt_tokens"] + c["completion_tokens"] for c in calls
        ),
        "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
    }
