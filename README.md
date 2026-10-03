# ai-agent-toolkit

Production-grade agent patterns using LangChain and LangGraph. This toolkit contains practical implementations of common agentic workflows, including RAG, autonomous tool use, and multi-agent orchestration.

## Agent Patterns

- **RAG Agent** (agents/rag_agent.py): Scalable retrieval-augmented generation for document-based QA.
- **Tool-Calling Agent** (agents/tool_calling_agent.py): Autonomous agent capable of using search, math, and file system tools.
- **Multi-Agent System** (agents/multi_agent.py): Directed acyclic graph (DAG) orchestration using Researcher, Writer, and Reviewer nodes.
- **Memory Agent** (agents/memory_agent.py): Stateful conversational agent with sliding-window history and auto-summarization.
- **Workflow Agent** (agents/workflow_agent.py): Dependency-ordered step runner with retries and an execution report.
- **Web Scraping Agent** (agents/web_scraping_agent.py): Fetches, parses and crawls pages with no LLM required.
- **QA Evaluation Agent** (agents/qa_evaluation_agent.py): Scores answers against a rubric (LLM-graded, with an offline heuristic fallback).

## Architecture

The project follows a standard modular structure for easy integration:

```text
ai-agent-toolkit/
├── agents/             # Core agent implementations
├── utils/              # Configuration and LLM factory wrappers
├── tests/              # Offline pytest suite (no API keys needed)
├── examples/           # Integration demos
├── docs/               # Architecture and design specs
├── pyproject.toml      # Packaging
└── requirements.txt    # Project dependencies
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/kagithamanoj/ai-agent-toolkit.git
cd ai-agent-toolkit

# Install the package (add ",search" to enable the Tavily web-search tool)
pip install -e ".[dev]"
# or, without packaging:  pip install -r requirements.txt

# Environment Setup
cp .env.example .env
# Configure OPENAI_API_KEY and TAVILY_API_KEY in .env
```

## Quick Start

Individual agents can be tested as standalone modules:

```bash
# Run RAG demo
python -m agents.rag_agent -q "Explain the core concepts of RAG"

# Run Tool-Calling demo
python -m agents.tool_calling_agent -q "What is 15% of 2340?"

# Run Multi-Agent demo
python -m agents.multi_agent -t "Advancements in LLM Orchestration"
```

The tool-calling agent logs token usage and estimated cost per LLM call, and the loop bound can be set from the command line:

```bash
python -m agents.tool_calling_agent -q "What is 15% of 2340?" --max-rounds 3
```

Or use an agent from Python:

```python
from agents.workflow_agent import WorkflowAgent

wf = WorkflowAgent("demo")
wf.add_step("load", lambda url: url.upper(), inputs=["url"])
wf.add_step("size", lambda load: len(load), depends_on=["load"])
print(wf.run(url="abc")["state"])   # {'load': 'ABC', 'size': 3}
```

For a comprehensive demonstration of all patterns, refer to `examples/quickstart.py`.

### What each module needs

| Module | Needs |
|---|---|
| `tool_calling_agent`, `multi_agent`, `memory_agent`, `rag_agent` | `OPENAI_API_KEY` (rag also uses OpenAI embeddings) |
| `tool_calling_agent` web search | `TAVILY_API_KEY` and the `search` extra |
| `qa_evaluation_agent` | `OPENAI_API_KEY` optional; falls back to offline heuristics |
| `web_scraping_agent`, `workflow_agent` | nothing (standard library only) |

Safety notes: the calculator tool evaluates an arithmetic-only syntax tree (no `eval`), and the file reader marks content that looks like a prompt-injection attempt as untrusted data. That scan is a heuristic warning, not a security boundary; do not expose the file reader to untrusted users.

## Testing

The test suite uses fake LLMs and mocked network calls, so it needs no API keys and no internet:

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v --cov=agents --cov=utils
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Release notes are in [CHANGELOG.md](CHANGELOG.md); to cite the software, see [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).

---

**Manoj Kumar Kagitha**
[GitHub](https://github.com/kagithamanoj)