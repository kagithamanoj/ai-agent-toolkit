from langchain_core.messages import AIMessage

from agents.summarize_agent import SummarizeAgent, chunk_text


class FakeLLM:
    """Returns scripted string replies in order, recording prompts.

    Defines __call__ so it can be piped into a LangChain chain
    (RunnableLambda wraps any callable).
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def invoke(self, prompt_value):
        self.seen.append(prompt_value)
        return AIMessage(content=self.replies.pop(0))

    def __call__(self, prompt_value):
        return self.invoke(prompt_value)


def make_agent(replies, **kwargs):
    fake = FakeLLM(replies)
    agent = SummarizeAgent(llm=fake, **kwargs)
    return agent, fake


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_and_blank_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunks_respect_size_and_dont_split_words():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=40, overlap=0)
    assert all(len(c) <= 40 for c in chunks)
    # every original word survives intact in at least one chunk
    joined = " ".join(chunks)
    for i in range(50):
        assert f"word{i}" in joined


def test_overlap_carries_text_between_chunks():
    text = "aaaa bbbb cccc dddd eeee ffff"
    chunks = chunk_text(text, chunk_size=10, overlap=5)
    assert len(chunks) > 1
    assert chunks[0][-5:] in chunks[1][:10]


def test_overlap_zero_packs_tightly():
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=7, overlap=0)
    assert len(chunks) > 1
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=-1)


# ── map / reduce ──────────────────────────────────────────────────────────────

def test_map_chunk_uses_instruction_and_chunk():
    agent, fake = make_agent(["MAP RESULT"])
    out = agent.map_chunk("some chunk text")
    assert out == "MAP RESULT"
    prompt_text = fake.seen[0].to_string()
    assert "some chunk text" in prompt_text
    assert "2-4 sentences" in prompt_text  # default map instruction present


def test_reduce_single_summary_skips_llm():
    agent, fake = make_agent([])
    assert agent.reduce(["only one"]) == "only one"
    assert fake.seen == []  # no LLM call needed


def test_reduce_merges_multiple_summaries():
    agent, fake = make_agent(["FINAL"])
    out = agent.reduce(["first part", "second part"])
    assert out == "FINAL"
    prompt_text = fake.seen[0].to_string()
    assert "first part" in prompt_text and "second part" in prompt_text
    assert "coherent summary" in prompt_text  # default reduce instruction present


def test_custom_instructions_are_used():
    agent, fake = make_agent(["X", "Y"], map_instruction="CUSTOM MAP", reduce_instruction="CUSTOM REDUCE")
    agent.map_chunk("chunk")
    assert "CUSTOM MAP" in fake.seen[0].to_string()
    agent.reduce(["a", "b"])
    assert "CUSTOM REDUCE" in fake.seen[1].to_string()


# ── end to end ────────────────────────────────────────────────────────────────

def test_summarize_empty_text():
    agent, _ = make_agent([])
    result = agent.summarize("   ")
    assert result == {
        "chunk_count": 0,
        "chunk_summaries": [],
        "final_summary": "",
        "input_chars": 0,
        "summary_chars": 0,
    }


def test_summarize_single_chunk():
    agent, fake = make_agent(["ONLY SUMMARY"])
    result = agent.summarize("short doc")
    assert result["chunk_count"] == 1
    assert result["chunk_summaries"] == ["ONLY SUMMARY"]
    assert result["final_summary"] == "ONLY SUMMARY"  # no reduce call
    assert result["input_chars"] == len("short doc")
    assert result["summary_chars"] == len("ONLY SUMMARY")
    assert len(fake.seen) == 1


def test_summarize_multi_chunk_map_reduce(capsys):
    text = " ".join(f"sentence{i} about topic{i}." for i in range(40))
    n_chunks = len(chunk_text(text, chunk_size=60, overlap=10))
    assert n_chunks > 1
    agent, fake = make_agent(
        [f"MAP {i}" for i in range(n_chunks)] + ["MERGED FINAL"],
        chunk_size=60,
        overlap=10,
    )
    result = agent.summarize(text)
    assert result["chunk_count"] == n_chunks
    assert result["chunk_summaries"] == [f"MAP {i}" for i in range(n_chunks)]
    assert result["final_summary"] == "MERGED FINAL"
    out = capsys.readouterr().out
    assert "Split document into" in out
    assert "map [1]" in out
    assert "reduce merging" in out
