"""Check that the LLM server (and optionally ENA and the encoder packages) work as the scripts expect.

usage: check_setup.py --model MODEL [--email you@example.org] [--encoders]

Checks, in order:
  1. the openai package is installed and OPENAI_BASE_URL is set
  2. the server answers /models and lists MODEL
  3. thinking can be switched off (chat_template_kwargs) -- no reasoning comes back
  4. thinking is on when asked, and a small thinking_token_budget is honoured
  5. one benchmark text with the production prompt gives parsable JSON with the expected places
  6. with --email: ENA answers a study query
  7. with --encoders: torch, transformers, gliner, gliner2, spacy are importable

Exits non-zero if any check fails; a WARN means the scripts run but results may differ from REPORT.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Model, complete  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
failed = False


def report(status, what, detail=""):
    global failed
    failed |= status == "FAIL"
    print(f"[{status:4}] {what}" + (f": {detail}" if detail else ""), flush=True)


def check_llm(model_name):
    try:
        import openai
    except ImportError:
        return report("FAIL", "openai package", "pip install -r requirements.txt")
    report("OK", "openai package", openai.__version__)
    if not os.environ.get("OPENAI_BASE_URL"):
        return report("FAIL", "OPENAI_BASE_URL", "not set (e.g. http://localhost:8000/v1)")
    report("OK", "OPENAI_BASE_URL", os.environ["OPENAI_BASE_URL"])
    if not os.environ.get("OPENAI_API_KEY"):
        report("WARN", "OPENAI_API_KEY", "not set (nor OPENAI_API_KEY_FILE); sending a dummy key (fine for vLLM without --api-key)")
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", timeout=300, max_retries=0)

    try:
        served = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        return report("FAIL", "server reachable", f"{type(e).__name__}: {str(e)[:200]}")
    if model_name not in served:
        return report("FAIL", f"model {model_name!r} listed", f"server lists {served}")
    report("OK", "model listed", model_name)

    def ask(model, system, user):
        try:
            return complete(client, model, system, user, "check")
        except Exception as e:
            report("FAIL", "request", f"{type(e).__name__}: {str(e)[:300]}")
            return None

    c = ask(Model(model_name, max_tokens=256), "Answer briefly.", "What is 17 + 25? Answer with the number only.")
    if c is None:
        return
    if c.reasoning_chars:
        report("WARN", "thinking off", f"{c.reasoning_chars} reasoning characters came back; the 'nothink' arms will think")
    elif "<think>" in c.text:
        report("WARN", "thinking off", "the answer contains <think>; start vLLM with a --reasoning-parser for this model")
    else:
        report("OK", "thinking off", repr(c.text.strip()[:40]))

    budget = 64
    c = ask(Model(model_name, max_tokens=2048, thinking=True, thinking_budget=budget), "Answer briefly.",
            "A train leaves at 9:47 and arrives at 13:05. How long is the trip? Think it through, then answer.")
    if c is None:
        return
    if not c.reasoning_chars:
        report("WARN", "thinking on", "no reasoning came back (no --reasoning-parser, or the model does not think)")
    elif c.reasoning_chars > budget * 8:
        report("WARN", "thinking budget", f"{c.reasoning_chars} reasoning characters for a {budget}-token budget;"
               " the server seems to ignore thinking_token_budget, so the think256/1k/4k arms are unbounded")
    else:
        report("OK", "thinking budget", f"{c.reasoning_chars} reasoning characters for a {budget}-token budget")

    from prompts import SYSTEM_SAMP, USER_SAMP, parse_entries
    texts = {r["id"]: r for r in map(json.loads, open(ROOT / "data/eval_texts.jsonl"))}
    gold = {r["id"]: r for r in map(json.loads, open(ROOT / "data/gold.jsonl"))}
    tid = "USER2"
    c = ask(Model(model_name, max_tokens=4096, thinking=True, thinking_budget=1024), SYSTEM_SAMP,
            USER_SAMP.format(text=texts[tid]["text"]))
    if c is None:
        return
    entries = parse_entries(c.text)
    if entries is None:
        return report("FAIL", "benchmark prompt", f"no parsable JSON in the answer: {c.text[-300:]!r}")
    got = {t.lower() for t, *_ in entries}
    missing = [p for p in gold[tid]["places"] if not any(p.lower() in g or g in p.lower() for g in got)]
    detail = f"{len(entries)} places in {c.latency_s:.1f} s: {sorted(t for t, *_ in entries)}"
    report("WARN" if missing else "OK", "benchmark prompt", detail + (f"; missed {missing}" if missing else ""))


def check_ena(email):
    from ena import fetch_studies
    try:
        rows = fetch_studies(["ERP021705"], email, tool="toponym-ner-check")
    except SystemExit as e:
        return report("FAIL", "ENA", str(e))
    except Exception as e:
        return report("FAIL", "ENA", f"{type(e).__name__}: {e}")
    report("OK" if rows else "FAIL", "ENA", rows[0]["study_title"][:60] if rows else "no record for ERP021705")


def check_encoders():
    import importlib
    for mod in ("torch", "transformers", "gliner", "gliner2", "spacy"):
        try:
            m = importlib.import_module(mod)
            report("OK", mod, getattr(m, "__version__", ""))
        except ImportError as e:
            report("FAIL", mod, f"{e}; pip install -r requirements-encoders.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="model name as the server lists it (or $LLM_MODEL)")
    ap.add_argument("--email", default=os.environ.get("ENA_EMAIL"), help="also check ENA, sending this contact address (or ENA_EMAIL)")
    ap.add_argument("--encoders", action="store_true", help="also check the encoder packages")
    a = ap.parse_args()
    if a.model:
        check_llm(a.model)
    else:
        report("SKIP", "LLM server", "no --model given")
    if a.email:
        check_ena(a.email)
    if a.encoders:
        check_encoders()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
