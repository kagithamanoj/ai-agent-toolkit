# Contributing to ai-agent-toolkit

Thanks for taking the time to contribute. This project is a set of
practical agent patterns built with LangChain and LangGraph, and
contributions that keep the patterns readable and tested are welcome.

## How to contribute

1. Fork the repo and create a branch for your change.
2. Keep the change focused. One pattern, one fix, or one doc update
   per pull request.
3. Add or update tests for any behavior change. Run the full test
   suite before opening the pull request.
4. Run the linter. The project uses ruff, and the code should pass
   clean before review.
5. Write a clear pull request description: what changed, why it was
   needed, and how you tested it.

## Reporting issues

Open an issue with steps to reproduce the problem, what you expected
to happen, and what actually happened. Include the pattern name
(for example, RAG agent or multi-agent system) and any error output.

## What fits here

- New agent patterns that follow the existing structure: a standalone
  module in `agents/` with its own tests.
- Improvements to existing patterns, docs, or examples.
- Fixes for bugs, broken tests, or outdated docs.

## What does not fit

- Changes that break existing tests without a clear reason.
- Features that depend on services with no offline test path.
- Large rewrites with no discussion first. Open an issue before
  starting anything big.

## Code style

Follow the style already in the repo. Short functions, plain names,
and comments only where the reason is not obvious from the code.
Keep the public surface small: if a helper is only used in one
module, keep it there.

## License

By contributing, you agree that your contribution will be licensed
under the MIT License, the same license as the rest of the project.
