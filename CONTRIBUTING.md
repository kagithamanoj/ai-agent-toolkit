# Contributing

Thanks for your interest in ai-agent-toolkit. Bug reports, fixes and new agent patterns are welcome.

## Reporting issues

Open an issue at <https://github.com/kagithamanoj/ai-agent-toolkit/issues> with the Python version, the module you were using, and a minimal reproduction. Never paste API keys.

## Development setup

```bash
git clone https://github.com/kagithamanoj/ai-agent-toolkit.git
cd ai-agent-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,search]"
```

## Before opening a pull request

```bash
ruff check .
python -m pytest tests/ -v
```

- Tests must run offline: use fake LLMs and mocked network calls, never real API keys.
- Keep each agent pattern in one self-contained module; avoid hidden framework magic.
- Add or update tests for any behavior change, and add a line to `CHANGELOG.md`.

## Support and maintenance

This is a single-maintainer project. Issues and pull requests are normally answered within two weeks. Security-sensitive reports can be sent privately to the maintainer via the GitHub profile contact.
