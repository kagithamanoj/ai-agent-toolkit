"""Shared fixtures. Every test runs offline: no API keys or network are needed."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Client constructors only need *a* key to exist; no request is ever sent.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
