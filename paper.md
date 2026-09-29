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

ai-agent-toolkit is a small open-source Python toolkit for building systems from large language model agents. It implements five agent patterns that show up again and again in production work: a tool-calling loop, a pipeline DAG, a supervisor/worker team, a RAG-grounded agent, and a stateful memory agent. Each pattern is one module, built on LangChain and LangGraph. The toolkit also ships the plumbing that production systems need from day one: per-run tracing with token counts, step and token budgets, guardrail hooks at the tool boundary, and tenant-keyed memory.

# Statement of need

Agent tooling clusters at two extremes. Demos wire a model to a few tools and stop. Platforms bring orchestration, policy, and observability with a heavy adoption cost. Teams shipping real systems need something between: working implementations of the standard patterns, small enough to read in an afternoon and adapt without a rewrite.

This toolkit grew out of production work on agentic AI platforms, where the same five patterns kept reappearing and the missing pieces were always the same: budgets, traces, and guardrail hooks. Most agent code we reviewed had none of the three. A runaway loop was one bad prompt away, nobody could reconstruct what an agent had done, and the tool boundary had no policy at all. ai-agent-toolkit bakes these in from the start so teams inherit them instead of rediscovering them the hard way. The design follows the same principle as our earlier work on API onboarding at scale: governance belongs in the plumbing, not in documentation [@kagitha2026ssrn].

# Features

**Tool-calling agent.** One agent, one loop: reason, call a tool, read the result, repeat until done or until the step budget runs out. Ships with search, math, and filesystem tools, in the spirit of the ReAct pattern [@yao2023react]. Every loop carries a step budget and a token budget, and each iteration is its own trace span, so a runaway loop is visible before the invoice arrives.

**Pipeline DAG.** Researcher, Writer, and Reviewer agents wired as LangGraph nodes and edges. The Reviewer works against explicit rejection criteria with a round limit, and failed drafts escalate to a human instead of looping forever.

**Supervisor/worker.** A supervisor decomposes a task, fans out to workers, and merges partial results, following the hierarchical pattern popularized by AutoGen [@wu2024autogen]. The merge logic tolerates partial failure: retries, quarantine, and honest partial summaries instead of invented completions.

**RAG-grounded agent.** A retriever paired with the generator, with citations on answers and source checks so restricted documents never leak to unauthorized users.

**Memory agent.** Sliding-window history plus rolling summaries, keyed by tenant and user, with retention rules on the summaries themselves.

**Cross-cutting.** Every run produces one trace with spans for plans, agent calls, tool calls, and guardrail decisions, with token usage attached. Guardrail hooks sit at three points: input screening, per-tool allowlist checks, and output screening. Deny by default, and log every denial.

# Design notes

The toolkit is deliberately boring. Explicit edges beat clever prompts: when the flow is code, the failure is findable. This mirrors the broader lesson from our work on deploying enterprise AI, where structure and measurement mattered more than model choice [@kagitha2024beyond; @kagitha2021cognitive]. The guardrail hooks apply the same check-before-execute discipline we previously used for flagging infrastructure-as-code vulnerabilities in CI/CD pipelines [@kagitha2021compliance].

# References
