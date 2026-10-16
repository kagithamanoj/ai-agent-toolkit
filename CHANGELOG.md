# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ReAct agent loop: `--max-rounds` CLI flag caps the reasoning loop
  (validated, at least 1; default 5).
- ReAct agent loop: streaming output. `run_agent_loop(stream=True)` prints
  tokens as they arrive through the LLM's stream API and merges the chunks
  into a single reply; `--stream` on the CLI.
- ReAct agent loop: per-call token usage and cost logging. New
  `utils/usage.py` pricing table, per-call records with printed lines and a
  final summary, and an optional `usage_log` parameter for programmatic
  access.
- ReAct agent loop: `--verbose` / `-v` tracing prints a per-round trace
  header, LLM step and tool call timings, and an end-of-run timing summary.
  Also available as the `verbose` agent config key.
- Per-tool call timeouts. Each tool call runs in a daemon thread via
  `invoke_with_timeout`; calls that run past the timeout raise
  `ToolTimeoutError` and are reported back to the model so it can adapt.
  Configurable with `tool_timeout` (loop parameter, `--tool-timeout` CLI
  flag, agent config key; default 30s).
- New `summarize` agent pattern: map-reduce over long documents.
- Tavily web-search tool: exponential-backoff retry (`_call_with_retry`,
  3 attempts with 1s/2s/4s backoff plus jitter).
- File-reader tool: prompt-injection guard. `scan_for_injection_markers`
  scans file content; suspicious files are returned inside untrusted-data
  delimiters with a SECURITY NOTICE so the model treats them as data, not
  instructions.
- YAML config file support for agent defaults (`utils/agent_config.py`,
  `--config` CLI flag). Precedence: CLI flag, then config file, then
  built-in defaults.
- Offline evaluation harness (`examples/eval_harness.py`) that runs all
  seven patterns against 21 fixed tasks with deterministic stub LLMs and a
  lexical fake embedding, then prints a comparison table. Needs no API keys
  and no internet.

### Docs

- README: mermaid architecture diagram showing the seven patterns.
- README: "Calculator tool" section documenting the AST evaluator's
  supported operators, math functions, constants, and what it refuses.
- Paper: new State of the field, Software design, and Research impact
  statement sections.

## [1.0.1] - 2026-09-29

### Docs

- Paper: AI usage disclosure.

## [1.0.0] - 2026-09-29

- Initial public release. Seven agent patterns (RAG, tool-calling,
  multi-agent pipeline, memory, QA evaluation, web scraping, workflow
  orchestration), safe AST-based calculator tool, offline test suite, and
  the JOSS paper draft.
