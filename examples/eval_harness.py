"""
Evaluation harness: run all seven agent patterns against a fixed task set and
print a comparison table.

Everything runs offline. Deterministic stub LLMs stand in for the real model
and fake embeddings stand in for the embedding service, so results are
reproducible with no API keys and no network access. The harness measures what
each pattern actually does (retrieval hit rate, real tool execution, memory
recall, workflow completion) rather than LLM answer quality.

Usage:
    python -m examples.eval_harness              # run all patterns
    python -m examples.eval_harness -p rag       # run one pattern
    python -m examples.eval_harness --list       # show the fixed task set
"""

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A dummy key keeps client constructors happy; nothing here ever sends a request.
os.environ.setdefault("OPENAI_API_KEY", "eval-harness-offline-key")

# The sandbox sets no_proxy/NO_PROXY with bracketed IPv6 tokens (e.g. "[::1]")
# that the vendored httpx URL parser cannot handle, which crashes OpenAI client
# construction even though no request is ever sent. The harness is fully
# offline, so drop just the bracketed tokens.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = ",".join(t for t in _val.split(",") if "[" not in t)

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

from agents import memory_agent, multi_agent, qa_evaluation_agent  # noqa: E402
from agents import rag_agent, tool_calling_agent, web_scraping_agent  # noqa: E402
from agents import workflow_agent  # noqa: E402


# ── Result types ────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Outcome of one fixed task."""
    pattern: str
    task: str
    passed: bool
    detail: str
    seconds: float


@dataclass
class PatternReport:
    """All results for one agent pattern plus summary metrics."""
    name: str
    results: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def elapsed(self) -> float:
        return round(sum(r.seconds for r in self.results), 2)


# ── Pattern 1: RAG agent ────────────────────────────────────────────────────────

_RAG_CORPUS = [
    ("handbook-1", "Acme Cloud is headquartered in Austin, Texas. The company was founded in 2019."),
    ("handbook-2", "The founder and CEO of Acme Cloud is Priya Nair."),
    ("handbook-3", "Acme Cloud pricing: the Starter plan costs $29 per month and includes 5 projects."),
    ("handbook-4", "Customers can reach support at support@acmecloud.example. Response time is under 4 hours."),
    ("handbook-5", "Acme Cloud's flagship product is Nimbus Deploy, a zero-downtime release tool."),
    ("blog-1", "The engineering blog covers Kubernetes, Terraform, and incident reviews."),
]

_RAG_TASKS = [
    ("Where is Acme Cloud headquartered?", "Austin"),
    ("What does the Starter plan cost?", "29"),
    ("Who founded Acme Cloud?", "Priya Nair"),
    ("What email should customers use for support?", "support@acmecloud.example"),
]


class _PromptRecorder(FakeListChatModel):
    """FakeListChatModel that records the prompt it was given."""

    seen_prompts: list

    def _generate(self, messages, *args, **kwargs):
        self.seen_prompts.append(messages[-1].content)
        return super()._generate(messages, *args, **kwargs)


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+(?:@[a-z0-9.]+)?", text.lower())


class _KeywordEmbedding(Embeddings):
    """Deterministic lexical embedding: one dimension per vocabulary word.

    Hash-based fake embeddings have no semantics, so retrieval quality with
    them would be pure luck. This embedding scores documents by shared
    vocabulary, which makes retrieval hit rate a meaningful, reproducible
    measure of the FAISS retrieval step while staying offline and
    deterministic.
    """

    def __init__(self, texts: list):
        vocab = set()
        for t in texts:
            vocab.update(_words(t))
        self.vocab = sorted(vocab)

    def _vec(self, text: str) -> list:
        words = set(_words(text))
        return [1.0 if w in words else 0.0 for w in self.vocab]

    def embed_documents(self, texts: list) -> list:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._vec(text)


def run_rag_tasks() -> PatternReport:
    report = PatternReport(name="rag")
    docs = [Document(page_content=text, metadata={"source": src}) for src, text in _RAG_CORPUS]
    store = FAISS.from_documents(docs, _KeywordEmbedding([t for _, t in _RAG_CORPUS]))

    for question, expected_keyword in _RAG_TASKS:
        start = time.time()
        llm = _PromptRecorder(responses=["(canned answer)"], seen_prompts=[])
        original = rag_agent.get_openai_llm
        rag_agent.get_openai_llm = lambda **kwargs: llm
        try:
            chain = rag_agent.build_rag_chain(vectorstore=store)
            answer = chain.invoke(question)
        finally:
            rag_agent.get_openai_llm = original
        prompt = llm.seen_prompts[-1] if llm.seen_prompts else ""
        top_hit = store.similarity_search(question, k=1)[0].metadata.get("source", "?")
        keyword_in_context = expected_keyword in prompt
        prompt_wired = "[Source:" in prompt and question in prompt
        passed = keyword_in_context and prompt_wired and bool(answer)
        report.results.append(TaskResult(
            pattern="rag",
            task=f"q: {question[:45]}",
            passed=passed,
            detail=(f"top hit '{top_hit}', expected '{expected_keyword}' in retrieved context: "
                    f"{'yes' if keyword_in_context else 'no'}, sources formatted into prompt: "
                    f"{'yes' if prompt_wired else 'no'}"),
            seconds=round(time.time() - start, 2),
        ))
    hits = sum(1 for r in report.results if r.passed)
    report.metrics = {"retrieval hit rate": f"{hits}/{report.total}"}
    return report


# ── Pattern 2: tool-calling agent ───────────────────────────────────────────────

_TOOL_TASKS = [
    ("What is 15% of 2340?", "15/100 * 2340", "351"),
    ("What is the square root of 144 plus 8?", "sqrt(144) + 8", "20"),
    ("Which is largest: 7, 42, or 13?", "max(7, 42, 13)", "42"),
]


class _CalculatorStubLLM:
    """First call: request the calculator tool. Second call: answer from its result."""

    def __init__(self, expression: str):
        self.expression = expression
        self.seen = []
        self._tool_requested = False

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self._tool_requested:
            self._tool_requested = True
            return AIMessage(content="", tool_calls=[{
                "name": "calculator",
                "args": {"expression": self.expression},
                "id": "call_1",
                "type": "tool_call",
            }])
        result = ""
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                result = m.content
                break
        match = re.search(r"Result:\s*([-\d.eE+]+)", result)
        value = match.group(1) if match else result
        return AIMessage(content=f"The calculator returned {value}.")


def run_tool_calling_tasks() -> PatternReport:
    report = PatternReport(name="tool-calling")
    for question, expression, expected in _TOOL_TASKS:
        start = time.time()
        stub = _CalculatorStubLLM(expression)
        original = tool_calling_agent.build_tool_agent
        tool_calling_agent.build_tool_agent = (
            lambda model=None, tools=None: (stub, [tool_calling_agent.calculator])
        )
        try:
            answer = tool_calling_agent.run_agent_loop(question, max_iterations=3)
        finally:
            tool_calling_agent.build_tool_agent = original
        tool_ran = any(
            isinstance(m, ToolMessage) and m.content.startswith("Result:")
            for convo in stub.seen for m in convo
        )
        passed = expected in answer and tool_ran
        report.results.append(TaskResult(
            pattern="tool-calling",
            task=f"{expression} = {expected}",
            passed=passed,
            detail=(f"calculator executed: {'yes' if tool_ran else 'no'}, "
                    f"answer contains '{expected}': {'yes' if expected in answer else 'no'}"),
            seconds=round(time.time() - start, 2),
        ))
    report.metrics = {"tool calls executed": f"{report.passed}/{report.total}"}
    return report


# ── Pattern 3: multi-agent ──────────────────────────────────────────────────────

_MULTI_TASKS = [
    "benefits of code reviews",
    "edge caching strategies",
]


class _MultiAgentStubLLM:
    """Deterministic stand-in for the researcher/writer/reviewer LLMs."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[1].content
        if "research analyst" in system:
            return SimpleNamespace(content=f"Research notes on the topic. {human} Key points: definition, trends, implications.")
        if "technical writer" in system:
            return SimpleNamespace(content=f"Draft article. {human[:120]} Conclusion: solid engineering practice.")
        if "senior editor" in system:
            return SimpleNamespace(content="APPROVED: draft meets the bar.")
        return SimpleNamespace(content="OK")


def run_multi_agent_tasks() -> PatternReport:
    report = PatternReport(name="multi-agent")
    original = multi_agent.get_llm
    multi_agent.get_llm = lambda *a, **k: _MultiAgentStubLLM()
    try:
        for topic in _MULTI_TASKS:
            start = time.time()
            state = multi_agent.run_sequential(topic)
            done = bool(state.get("final_output"))
            topic_covered = topic.split()[0] in state.get("final_output", "")
            passed = done and topic_covered
            report.results.append(TaskResult(
                pattern="multi-agent",
                task=f"topic: {topic}",
                passed=passed,
                detail=(f"researcher/writer/reviewer ran, {state.get('iteration', 0)} review round(s), "
                        f"final output {len(state.get('final_output', ''))} chars"),
                seconds=round(time.time() - start, 2),
            ))
    finally:
        multi_agent.get_llm = original
    report.metrics = {"topics drafted end to end": f"{report.passed}/{report.total}"}
    return report


# ── Pattern 4: memory agent ─────────────────────────────────────────────────────

class _MemoryChainStub:
    """Deterministic chain: answers from the conversation history it is given."""

    def invoke(self, payload):
        history = payload.get("history", [])
        user_input = payload.get("input", "")

        def recalled(pattern):
            for msg in reversed(history):
                m = re.search(pattern, getattr(msg, "content", ""), re.I)
                if m:
                    return m.group(1).strip().rstrip(".")
            return None

        if m := re.search(r"my name is (\w+)", user_input, re.I):
            return f"Nice to meet you, {m.group(1).title()}."
        if re.search(r"what is my name", user_input, re.I):
            name = recalled(r"my name is (\w+)")
            return f"Your name is {name.title()}." if name else "You have not told me your name."
        if m := re.search(r"i live in ([\w ]+)\.?\s*$", user_input.strip(), re.I):
            return f"{m.group(1).strip().title()} sounds nice."
        if re.search(r"where do i live", user_input, re.I):
            city = recalled(r"i live in ([\w ]+)\.?\s*$")
            return f"You live in {city.title()}." if city else "You have not told me where you live."
        return f"Echo: {user_input}"


def run_memory_tasks() -> PatternReport:
    report = PatternReport(name="memory")

    start = time.time()
    agent = memory_agent.MemoryAgent()
    agent.chain = _MemoryChainStub()
    agent.chat("My name is Ada.")
    reply = agent.chat("What is my name?")
    passed = "Ada" in reply
    report.results.append(TaskResult(
        pattern="memory", task="recall name across turns", passed=passed,
        detail=f"second-turn reply: '{reply}'",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    agent2 = memory_agent.MemoryAgent()
    agent2.chain = _MemoryChainStub()
    agent2.chat("I live in Berlin.")
    reply2 = agent2.chat("Where do I live?")
    passed2 = "Berlin" in reply2
    report.results.append(TaskResult(
        pattern="memory", task="recall city across turns", passed=passed2,
        detail=f"second-turn reply: '{reply2}'",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    agent3 = memory_agent.MemoryAgent()
    agent3.chain = _MemoryChainStub()
    agent3.chat("hello")
    passed3 = agent3.history_length == 2
    report.results.append(TaskResult(
        pattern="memory", task="history bookkeeping", passed=passed3,
        detail=f"history_length after one turn: {agent3.history_length} (expected 2)",
        seconds=round(time.time() - start, 2)))

    recalls = sum(1 for r in report.results[:2] if r.passed)
    report.metrics = {"cross-turn recall": f"{recalls}/2"}
    return report


# ── Pattern 5: QA evaluation agent ──────────────────────────────────────────────

_QA_TASKS = [
    ("thorough answer scores well",
     "What is machine learning?",
     "Machine learning lets systems learn from data. It has several types.\n"
     "- supervised\n- unsupervised. Is it useful? Yes, widely used in production!",
     "high"),
    ("one-liner scores poorly",
     "What is machine learning?",
     "Yes.",
     "low"),
]


def run_qa_eval_tasks() -> PatternReport:
    report = PatternReport(name="qa-eval")
    scores = {}
    for task_id, question, answer, expectation in _QA_TASKS:
        start = time.time()
        evaluator = qa_evaluation_agent.QAEvaluator()
        evaluator.api_key = None  # force the offline heuristic path
        result = evaluator.evaluate(question, answer)
        avg = result["average_score"]
        grade = result["grade"]
        if expectation == "high":
            passed = grade in ("A", "B")
        else:
            passed = grade in ("C", "D", "F")
        scores[task_id] = avg
        report.results.append(TaskResult(
            pattern="qa-eval", task=task_id, passed=passed,
            detail=f"avg {avg:.2f}/5, grade {grade} (expected {expectation})",
            seconds=round(time.time() - start, 2)))
    discriminates = scores["thorough answer scores well"] >= scores["one-liner scores poorly"]
    start = time.time()
    report.results.append(TaskResult(
        pattern="qa-eval", task="rubric discriminates quality", passed=discriminates,
        detail=(f"thorough {scores['thorough answer scores well']:.2f} vs "
                f"one-liner {scores['one-liner scores poorly']:.2f}"),
        seconds=round(time.time() - start, 2)))
    avg_all = sum(scores.values()) / len(scores)
    report.metrics = {"avg rubric score": f"{avg_all:.2f}/5"}
    return report


# ── Pattern 6: web scraping agent ───────────────────────────────────────────────

_SCRAPE_HTML = """<!DOCTYPE html>
<html><head>
<title>Acme Cloud Docs</title>
<meta name="description" content="Deployment guides for Acme Cloud.">
<meta property="og:title" content="Acme Docs">
</head><body>
<script>var secret_token = "abc123";</script>
<style>body { color: red; }</style>
<h1>Getting Started</h1>
<p>Welcome to the Acme Cloud documentation hub.</p>
<h2>Install the CLI</h2>
<p>Run the installer, then authenticate.</p>
<a href="/docs/install">Install guide</a>
<a href="https://example.com/status">Status page</a>
</body></html>"""


def run_web_scrape_tasks() -> PatternReport:
    report = PatternReport(name="web-scrape")
    agent = web_scraping_agent.WebScrapingAgent()

    start = time.time()
    text = agent.extract_text(_SCRAPE_HTML)
    passed = "Welcome to the Acme Cloud documentation hub." in text and "secret_token" not in text
    report.results.append(TaskResult(
        pattern="web-scrape", task="extract_text strips scripts", passed=passed,
        detail=f"{len(text)} chars extracted, script content excluded: {'yes' if 'secret_token' not in text else 'no'}",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    meta = agent.extract_metadata(_SCRAPE_HTML)
    passed = meta.get("title") == "Acme Cloud Docs" and meta.get("description") == "Deployment guides for Acme Cloud."
    report.results.append(TaskResult(
        pattern="web-scrape", task="extract_metadata", passed=passed,
        detail=f"title='{meta.get('title')}', description present: {'yes' if meta.get('description') else 'no'}",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    headings = agent.extract_headings(_SCRAPE_HTML)
    passed = len(headings) == 2 and headings[0] == {"level": 1, "text": "Getting Started"}
    report.results.append(TaskResult(
        pattern="web-scrape", task="extract_headings", passed=passed,
        detail=f"{len(headings)} headings, first: {headings[0] if headings else None}",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    links = agent.extract_links(_SCRAPE_HTML, base_url="https://docs.acme.example")
    passed = (len(links) == 2
              and links[0]["url"] == "https://docs.acme.example/docs/install"
              and links[1]["url"] == "https://example.com/status")
    report.results.append(TaskResult(
        pattern="web-scrape", task="extract_links resolves relative urls", passed=passed,
        detail=f"{len(links)} links: {[l['url'] for l in links]}",
        seconds=round(time.time() - start, 2)))

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(_SCRAPE_HTML)
        url = page.as_uri()
        first = agent.fetch(url)
        second = agent.fetch(url)
    passed = "<title>Acme Cloud Docs</title>" in first and first == second and url in agent.session_cache
    report.results.append(TaskResult(
        pattern="web-scrape", task="fetch reads local file and caches", passed=passed,
        detail=f"fetched {len(first)} chars, cache hit on second call: {'yes' if url in agent.session_cache else 'no'}",
        seconds=round(time.time() - start, 2)))

    report.metrics = {"extraction checks passed": f"{report.passed}/{report.total}"}
    return report


# ── Pattern 7: workflow agent ───────────────────────────────────────────────────

def run_workflow_tasks() -> PatternReport:
    report = PatternReport(name="workflow")
    start = time.time()

    attempts = {"flaky": 0}

    def flaky_step():
        attempts["flaky"] += 1
        if attempts["flaky"] == 1:
            raise ValueError("simulated transient failure")
        return "recovered"

    wf = workflow_agent.WorkflowAgent("eval-demo")
    wf.add_step("load", lambda text: text, inputs=["text"])
    wf.add_step("count", lambda load: len(load.split()), depends_on=["load"])
    wf.add_step("flaky", flaky_step)
    wf.add_step("report", lambda count, flaky: f"{count} words; flaky={flaky}",
                depends_on=["count", "flaky"])
    result = wf.run(text="alpha beta gamma delta")

    ok = (result["failed"] == 0
          and result["state"].get("report") == "4 words; flaky=recovered"
          and attempts["flaky"] == 2)
    report.results.append(TaskResult(
        pattern="workflow", task="4-step workflow with dependency order and retry", passed=ok,
        detail=(f"completed {result['completed']}/{result['total_steps']} steps, "
                f"flaky step attempts: {attempts['flaky']}, "
                f"report: '{result['state'].get('report')}'"),
        seconds=round(time.time() - start, 2)))
    report.metrics = {"steps completed": f"{result['completed']}/{result['total_steps']}",
                      "retries exercised": "1"}
    return report


# ── Runner ──────────────────────────────────────────────────────────────────────

_PATTERN_RUNNERS = {
    "rag": run_rag_tasks,
    "tool-calling": run_tool_calling_tasks,
    "multi-agent": run_multi_agent_tasks,
    "memory": run_memory_tasks,
    "qa-eval": run_qa_eval_tasks,
    "web-scrape": run_web_scrape_tasks,
    "workflow": run_workflow_tasks,
}


def run_all(pattern: str = None) -> dict:
    """Run the harness for one pattern or all of them. Returns name -> PatternReport."""
    selected = [pattern] if pattern else list(_PATTERN_RUNNERS)
    reports = {}
    for name in selected:
        runner = _PATTERN_RUNNERS[name]
        print(f"\n=== {name} ===")
        try:
            reports[name] = runner()
        except Exception as e:  # one broken pattern must not kill the table
            failed = PatternReport(name=name)
            failed.results.append(TaskResult(
                pattern=name, task="harness run", passed=False,
                detail=f"runner raised {type(e).__name__}: {e}", seconds=0.0))
            reports[name] = failed
        for r in reports[name].results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.task} ({r.seconds}s)\n        {r.detail}")
    return reports


def print_table(reports: dict) -> None:
    print("\n" + "=" * 78)
    print("EVAL HARNESS COMPARISON TABLE (offline, deterministic stubs)")
    print("=" * 78)
    header = f"{'pattern':<14}{'tasks':>7}{'passed':>8}{'success':>9}{'time(s)':>9}  key metric"
    print(header)
    print("-" * 78)
    for name, rep in reports.items():
        pct = f"{100 * rep.passed // rep.total}%" if rep.total else "n/a"
        metrics = ", ".join(f"{k} {v}" for k, v in rep.metrics.items())
        print(f"{name:<14}{rep.total:>7}{rep.passed:>8}{pct:>9}{rep.elapsed:>9.2f}  {metrics}")
    print("-" * 78)
    total_tasks = sum(r.total for r in reports.values())
    total_passed = sum(r.passed for r in reports.values())
    print(f"{'TOTAL':<14}{total_tasks:>7}{total_passed:>8}")
    print("=" * 78)


def list_tasks() -> None:
    print("Fixed task set (one run evaluates every task):\n")
    for name, runner in _PATTERN_RUNNERS.items():
        print(f"{name}:")
        if name == "rag":
            for q, kw in _RAG_TASKS:
                print(f"  - answer '{q}' (expect '{kw}' in retrieved context)")
        elif name == "tool-calling":
            for q, expr, expected in _TOOL_TASKS:
                print(f"  - '{q}' via calculator({expr}) (expect {expected})")
        elif name == "multi-agent":
            for topic in _MULTI_TASKS:
                print(f"  - researcher/writer/reviewer pipeline on '{topic}'")
        elif name == "memory":
            print("  - recall a name across two turns")
            print("  - recall a city across two turns")
            print("  - history bookkeeping (history_length == 2 after one turn)")
        elif name == "qa-eval":
            print("  - thorough answer should score high (A/B)")
            print("  - one-liner should score low (C/D/F)")
            print("  - rubric must rank the thorough answer at least as high")
        elif name == "web-scrape":
            print("  - extract_text strips scripts, keeps body text")
            print("  - extract_metadata returns title and description")
            print("  - extract_headings returns both headings")
            print("  - extract_links resolves relative urls against base_url")
            print("  - fetch reads a local file and caches it")
        elif name == "workflow":
            print("  - 4-step workflow: dependency order, one retry, final report")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run all seven agent patterns against a fixed task set.")
    parser.add_argument("-p", "--pattern", choices=sorted(_PATTERN_RUNNERS),
                        help="run only this pattern")
    parser.add_argument("--list", action="store_true", help="show the fixed task set and exit")
    args = parser.parse_args(argv)

    if args.list:
        list_tasks()
        return 0

    reports = run_all(args.pattern)
    print_table(reports)
    failed = [f"{name}: {r.task}" for name, rep in reports.items()
              for r in rep.results if not r.passed]
    if failed:
        print(f"\n{len(failed)} task(s) failed:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nAll tasks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
