# 🤖 AI Agent Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green)](https://docs.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-🦜🕸️-purple)](https://langchain-ai.github.io/langgraph/)

> A collection of production-ready AI agent patterns — RAG, tool-calling, multi-agent orchestration, and conversational memory — built with LangChain and LangGraph.

## 🎯 What's Inside

| Agent Pattern | File | Description |
|-------------|------|-------------|
| 🔍 **RAG Agent** | `agents/rag_agent.py` | Retrieval-augmented generation — answers questions from your documents |
| 🔧 **Tool-Calling Agent** | `agents/tool_calling_agent.py` | Agent that uses tools (web search, calculator, file reader) |
| 👥 **Multi-Agent System** | `agents/multi_agent.py` | Researcher → Writer → Reviewer pipeline with LangGraph |
| 💬 **Memory Agent** | `agents/memory_agent.py` | Conversational agent with sliding-window memory |

## 🏗️ Architecture

```
ai-agent-toolkit/
├── agents/
│   ├── rag_agent.py              # Document Q&A with vector search
│   ├── tool_calling_agent.py     # Agent with external tools
│   ├── multi_agent.py            # Multi-agent orchestration
│   └── memory_agent.py           # Chat with conversation memory
├── utils/
│   └── llm_config.py             # Centralized LLM configuration
├── examples/
│   └── quickstart.py             # Quick demos for all patterns
├── docs/
│   └── architecture.md           # Detailed design documentation
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kagithamanoj/ai-agent-toolkit.git
cd ai-agent-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
# RAG Agent — Ask questions about documents
python -m agents.rag_agent -q "What is RAG?" 

# Tool-Calling Agent — Use tools to find answers
python -m agents.tool_calling_agent -q "What is sqrt(144) + 25?"

# Multi-Agent — Generate a researched article
python -m agents.multi_agent -t "The future of AI agents"

# Memory Agent — Interactive chat with memory
python -m agents.memory_agent

# Run all demos
python examples/quickstart.py
```

## 🔍 RAG Agent

Answers questions by retrieving relevant context from your documents.

```python
from agents.rag_agent import build_rag_chain, load_documents_from_urls

# Index web pages
docs = load_documents_from_urls(["https://docs.langchain.com/docs/"])
chain = build_rag_chain(documents=docs)

answer = chain.invoke("How do I create a chain?")
```

**Supports**: Text files, PDFs, web pages, directories

## 🔧 Tool-Calling Agent

Agent that autonomously decides which tools to use.

```python
from agents.tool_calling_agent import run_agent_loop

# The agent will use the calculator tool automatically
answer = run_agent_loop("What's 15% of $2,340?")

# Uses web search when it needs current info
answer = run_agent_loop("What are the latest AI developments?")
```

**Built-in tools**: Calculator, Web Search (Tavily), File Reader, DateTime

## 👥 Multi-Agent System

Three specialized agents collaborating via LangGraph:

```
Researcher → Writer → Reviewer → (approve or revise) → Final Output
```

```python
from agents.multi_agent import build_multi_agent_graph

graph = build_multi_agent_graph()
result = graph.invoke({"topic": "AI in healthcare", ...})
```

## 💬 Memory Agent

Interactive chatbot that remembers the conversation:

```python
from agents.memory_agent import MemoryAgent

agent = MemoryAgent()
agent.chat("My name is Manoj")     # → "Nice to meet you, Manoj!"
agent.chat("What's my name?")      # → "Your name is Manoj!"
```

**Memory**: Sliding window (20 messages) + auto-summarization

## ⚙️ Configuration

All credentials are loaded from environment variables via `utils/llm_config.py`:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `TAVILY_API_KEY` | Optional | For web search tool |
| `OLLAMA_BASE_URL` | Optional | Local Ollama endpoint |

**Switch to local models** by using `get_ollama_llm()` instead of `get_openai_llm()`.

## 📖 Documentation

See [docs/architecture.md](docs/architecture.md) for detailed design patterns, flow diagrams, and customization guides.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new agent patterns
- Add new tools for the tool-calling agent
- Improve documentation
- Report issues

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with** 🦜 [LangChain](https://docs.langchain.com) + 🕸️ [LangGraph](https://langchain-ai.github.io/langgraph/) by [Manoj Kumar Kagitha](https://github.com/kagithamanoj)