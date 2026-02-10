# Architecture & Design Patterns

## Overview

The AI Agent Toolkit is organized around four distinct agent patterns, each solving a different class of problem:

```
ai-agent-toolkit/
├── agents/
│   ├── rag_agent.py          ← Retrieval-Augmented Generation
│   ├── tool_calling_agent.py ← External tool integration
│   ├── multi_agent.py        ← Multi-agent orchestration
│   └── memory_agent.py       ← Conversational memory
├── utils/
│   └── llm_config.py         ← Centralized LLM configuration
└── examples/
    └── quickstart.py          ← Quick demos for all patterns
```

## Agent Patterns

### 1. RAG Agent (`rag_agent.py`)

**When to use**: You have a knowledge base (documents, PDFs, web pages) and want an AI that answers questions from that specific content.

```
User Question → Retrieve Relevant Chunks → Augment Prompt → Generate Answer
                      ↑
              Vector Store (FAISS)
```

**Key components**:
- Document loaders (directory, URL, PDF)
- Recursive text splitter (1000 char chunks, 200 overlap)
- FAISS vector store with similarity search (top-4)
- Grounded prompt that forces source citation

### 2. Tool-Calling Agent (`tool_calling_agent.py`)

**When to use**: You need an AI that can take actions — search the web, do calculations, read files, etc.

```
User Question → LLM Decides Tool → Execute Tool → LLM Interprets Result
                     ↑                                      │
                     └──────────── Loop (max 5 rounds) ─────┘
```

**Built-in tools**:
| Tool | Description |
|------|-------------|
| `calculator` | Safe math evaluation |
| `web_search` | Tavily web search |
| `read_file` | Local file reading (50KB max) |
| `current_datetime` | Current date/time |

### 3. Multi-Agent System (`multi_agent.py`)

**When to use**: Complex tasks that benefit from specialized roles working together.

```
         ┌──────────┐     ┌────────┐     ┌──────────┐
Topic → │ Researcher │ → │ Writer  │ → │ Reviewer  │ → Output
         └──────────┘     └────────┘     └──────┬───┘
                               ↑               │
                               └── Revise ──────┘
                            (if not approved)
```

**Design**: Uses LangGraph's `StateGraph` for orchestration, with a conditional edge that loops the draft back to the Writer if the Reviewer doesn't approve.

### 4. Memory Agent (`memory_agent.py`)

**When to use**: Interactive chatbots that need to remember conversation context.

```
User Input → [History + Summary] → LLM → Response
                    ↑                        │
              Memory Store ←─────── Save ────┘
```

**Memory strategy**: Sliding window (20 messages) + auto-summarization of older messages.

## Configuration

All LLM access is centralized in `utils/llm_config.py`:
- **No hardcoded keys** — everything via environment variables
- **Provider-agnostic** — swap between OpenAI and Ollama
- **Consistent interface** — same function signatures across providers
