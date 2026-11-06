"""Load settings from the repository's .env file (KEY=value lines) into the environment.

Variables already set in the environment win. If OPENAI_API_KEY is unset but OPENAI_API_KEY_FILE
names a file, the key is read from that file, so a key kept elsewhere need not be copied.
Imported for its side effect by llm.py and ena.py.
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load(path=ENV_FILE):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            os.environ.setdefault(key, os.path.expanduser(value))
    key_file = os.environ.get("OPENAI_API_KEY_FILE")
    if not os.environ.get("OPENAI_API_KEY") and key_file:
        os.environ["OPENAI_API_KEY"] = Path(key_file).expanduser().read_text().strip()


load()
