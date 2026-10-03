# Changelog

## 1.0.2 - 2026-10-03
- Untrack committed `__pycache__` / `.pytest_cache` files.
- Add `pyproject.toml` (installable package), `CONTRIBUTING.md`, `CITATION.cff`, this changelog.
- Document token/cost logging, `--max-rounds`, search retry and the file-reader injection scan in the README.
- CI reports test coverage.

## 1.0.1 - 2026-09-29
- Add AI usage disclosure to the JOSS paper.

## 1.0.0 - 2026-09-29
- First tagged release: seven agent modules, offline pytest suite, MIT license, JOSS paper.
- Calculator tool now uses an AST-based evaluator instead of `eval`.
- Fix unreachable branch in the QA evaluator's completeness heuristic.
