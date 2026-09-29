---
title: 'ai-agent-toolkit: A Small, Readable Toolkit for Building LLM Agent Systems'
tags:
  - Python
  - large language models
  - AI agents
  - LangChain
  - LangGraph
  - retrieval-augmented generation
authors:
  - name: Manoj Kumar Kagitha
    orcid: 0009-0001-3484-3386
    affiliation: 1
affiliations:
  - name: Eficens Systems Inc., Dallas, TX, USA
    index: 1
date: 29 September 2026
bibliography: paper.bib
---

# Summary

ai-agent-toolkit is a small open-source Python toolkit of working implementations of common LLM agent patterns. It contains seven modules, each one file, each runnable on its own: a tool-calling loop, a retrieval-augmented generation chain, a Researcher/Writer/Reviewer multi-agent pipeline, a conversational memory agent, an LLM-as-judge QA evaluator, a web scraping helper, and a dependency-ordered workflow runner. Everything is built on LangChain, with LangGraph for the multi-agent pipeline, and every module ships with a command-line entry point.

# Statement of need

Agent code tends to live at two extremes. Demos wire a model to a couple of tools and stop before anything hard shows up. Platforms bring orchestration, policy, and observability with a heavy adoption cost. Teams building real systems usually need something in between: implementations of the standard patterns that are small enough to read in an afternoon, run as-is, and adapt without a rewrite.

This toolkit is that middle ground. Each pattern is one module with no hidden framework magic. The control flow is plain code: loops with explicit bounds, steps with declared dependencies, state passed through typed dicts. When something fails, the failure is findable. That emphasis on boring, inspectable plumbing reflects the broader lesson of our work on production AI systems, where the engineering around the model mattered more than the model itself [@kagitha2024beyond].

# State of the field

The ReAct pattern showed that interleaving reasoning traces with tool calls gives language models a general way to act [@yao2023react]. Since then the ecosystem has split into two layers. Frameworks like LangChain and LangGraph provide the primitives: chat models, tool binding, and graph-based orchestration. Multi-agent frameworks such as AutoGen [@wu2024autogen] build full conversation programming models on top, aimed at teams building complex agent systems. What is missing between the framework and the application is the worked example that survives contact with production: bounded loops instead of open-ended ones, typed state instead of dictionaries of anything, fallback paths when an optional dependency is absent. ai-agent-toolkit sits in that gap. It uses the same frameworks rather than competing with them, and its contribution is curation and restraint: seven standard patterns implemented small enough to read and solid enough to run.

# Features

**Tool-calling agent.** A ReAct-style loop [@yao2023react]: the model reasons, calls a tool, reads the result, and repeats, for at most five rounds. It ships with four tools: a calculator that safely evaluates math expressions, a Tavily web search, a local file reader capped at 50 KB, and a clock. Unknown tool calls are reported instead of crashing the loop.

**RAG agent.** Loads documents from a text directory, a list of URLs, or a PDF, splits them into overlapping chunks, and indexes them in a vector store. Retrieval is top-4 similarity search. The prompt restricts the model to the retrieved context, and each chunk carries a source label, so answers stay grounded in the documents provided.

**Multi-agent pipeline.** Researcher, Writer, and Reviewer agents wired as LangGraph nodes with a conditional edge: the Reviewer approves the draft or sends it back to the Writer with feedback, for at most two revision rounds. Shared state is a typed dict, and a sequential fallback runs the same three steps if LangGraph is not installed.

**Memory agent.** A conversational agent with a sliding window over the last twenty messages. Older messages roll into a running summary buffer that stays in the prompt. An interactive CLI supports resetting memory and checking message counts mid-chat.

**QA evaluator.** Scores generated answers against a five-dimension rubric (correctness, completeness, relevance, clarity, safety) using an LLM judge, with a heuristic fallback when no API key is set. It assigns letter grades, explains low scores with recommendations, and can batch-evaluate test suites with aggregate pass rates.

**Web scraping agent.** Fetches pages with an in-memory cache and extracts clean text from HTML, as a lightweight input stage for the other agents.

**Workflow agent.** Chains plain functions into a dependency-ordered workflow: topological sort decides the run order, each step retries up to twice on failure, results flow through shared state, and every run returns a report with per-step timing and status plus a text diagram of the plan.

# Software design

The design rule is one pattern, one file, no hidden machinery. Each of the seven modules is self-contained: its control flow is visible in the source, its loop bounds are explicit constants, and its state is a typed dict that a reader can inspect. Modules do not import each other. The multi-agent pipeline uses LangGraph nodes and a conditional edge, with a sequential fallback that runs the same three steps if LangGraph is not installed, so the pipeline reads as two versions of the same idea. Every module has a command-line entry point that runs the pattern end to end with sensible defaults, which doubles as a smoke test. The test suite exercises each module, including the calculator tool's expression evaluator, which parses and evaluates math on an abstract syntax tree rather than executing arbitrary code. Dependencies are limited to the LangChain and LangGraph packages plus a handful of small utilities, and the README documents exactly what each module needs.

# Research impact statement

The intended impact is practical and educational. Students learning agent patterns get implementations that run without a platform to install; practitioners evaluating whether an agent fits their problem get code they can read in an afternoon and adapt without a rewrite. The toolkit also serves as a companion to longer-form writing on production AI systems: the patterns here are the small-scale counterparts of the architectures discussed in our work on enterprise AI deployment [@kagitha2024beyond] and on the engineering of AI-enabled delivery pipelines [@kagitha2021cognitive]. By keeping each pattern minimal and its behavior explicit, the toolkit aims to be a reference that ages well: the frameworks will change, and the patterns, stated plainly, will transfer.

# AI usage disclosure

Generative AI tools assisted in drafting this paper and in developing parts of the software, including code generation, refactoring, and copy-editing. The author reviewed, edited, and validated all AI-assisted outputs and made the core design decisions.

# References
