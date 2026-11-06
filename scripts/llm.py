"""A minimal client for an OpenAI-compatible chat endpoint serving an open-weights model (e.g. vLLM).

Configuration comes from the environment, as for the openai package itself:
  OPENAI_BASE_URL   the server, e.g. http://localhost:8000/v1
  OPENAI_API_KEY    its key (vLLM without --api-key accepts any value), or OPENAI_API_KEY_FILE
These can also be set in the repository's .env file (see .env.example).

Thinking is controlled the way vLLM's Qwen chat templates expect it:
  - off:      chat_template_kwargs={"enable_thinking": False}
  - budget:   thinking_token_budget=N (a vLLM extension; other servers may ignore or reject it)
Run scripts/check_setup.py to see whether a server honours both.
"""
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import env  # noqa: F401  (loads .env)

RETRY_DELAYS_S = (30, 60, 120, 240)


@dataclass(frozen=True)
class Model:
    name: str                              # the model name the server lists under /v1/models
    max_tokens: int = 4096                 # thinking and answer share this limit
    thinking: bool = False
    thinking_budget: Optional[int] = None  # tokens; None = no limit
    json_schema: Optional[Dict[str, Any]] = None  # constrain the answer to this schema
    extra: Dict[str, Any] = field(default_factory=dict)  # anything else for the request body

    def request(self, system: str, user: str) -> Dict[str, Any]:
        body: Dict[str, Any] = dict(self.extra)
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.thinking_budget is not None:
            body["thinking_token_budget"] = self.thinking_budget
        params: Dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": body,
        }
        if self.json_schema is not None:
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "strict": True, "schema": self.json_schema}}
        return params


@dataclass
class Completion:
    text: str
    reasoning: str
    stop_reason: Optional[str]
    output_tokens: int
    latency_s: float
    attempts: int

    @property
    def reasoning_chars(self) -> int:
        return len(self.reasoning)


def make_client(timeout: float = 1800.0):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("the openai package is missing: pip install -r requirements.txt")
    if not os.environ.get("OPENAI_BASE_URL"):
        sys.exit("set OPENAI_BASE_URL to your OpenAI-compatible server, e.g. http://localhost:8000/v1")
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", timeout=timeout, max_retries=0)


def _retryable(exc: Exception) -> bool:
    import openai
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def complete(client, model: Model, system: str, user: str, label: str = "", pause_s: float = 0.0) -> Completion:
    """Stream one answer, retrying on 429, 5xx and connection errors; then sleep pause_s."""
    try:
        return _complete(client, model, system, user, label)
    finally:
        time.sleep(pause_s)


def _complete(client, model, system, user, label) -> Completion:
    attempt = 0
    while True:
        attempt += 1
        started = time.monotonic()
        text, reasoning, stop, usage = [], [], None, None
        try:
            for chunk in client.chat.completions.create(**model.request(system, user)):
                if chunk.usage:
                    usage = chunk.usage
                for choice in chunk.choices or []:
                    stop = choice.finish_reason or stop
                    delta = choice.delta
                    if delta is None:
                        continue
                    if delta.content:
                        text.append(delta.content)
                    # vLLM streams the thinking beside the answer, under one of these names
                    extra = delta.model_extra or {}
                    reasoning.append(extra.get("reasoning_content") or extra.get("reasoning") or "")
            break
        except Exception as exc:
            if not _retryable(exc) or attempt > len(RETRY_DELAYS_S):
                raise
            delay = RETRY_DELAYS_S[attempt - 1] * random.uniform(1.0, 1.2)
            print(f"    [{label}] {type(exc).__name__}: {str(exc)[:200]}; retry in {delay:.0f} s", file=sys.stderr)
            time.sleep(delay)
    return Completion(text="".join(text), reasoning="".join(reasoning), stop_reason=stop,
                      output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                      latency_s=time.monotonic() - started, attempts=attempt)
