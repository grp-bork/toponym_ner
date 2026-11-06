"""Ask an LLM on an OpenAI-compatible server for the place names in each evaluation text.

usage: run_llm.py --model MODEL [--name NAME] [--pause S] ARM [ARM ...]

Each arm writes preds/<NAME>_<ARM>.jsonl (NAME defaults to the model name after its last "/"),
one row per text, and skips texts already answered there, so a rerun resumes. For repeat runs of
the same arm, give each a different --name.

Arms (see REPORT.md):
  nothink, think256, think1k, think4k, think_unlimited   place names only, at several thinking budgets
  nothink_schema                                          no thinking, answer held to a JSON schema
  nothink_corr, think1k_corr                              plus a corrected spelling per place
  think1k_corr_samp                                       plus a "sampled" flag per place

Set OPENAI_BASE_URL and OPENAI_API_KEY first; scripts/check_setup.py tests the server.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Model, complete, make_client  # noqa: E402
from prompts import (SCHEMA, SYSTEM, SYSTEM_CORR, SYSTEM_SAMP, USER, USER_CORR, USER_SAMP,  # noqa: E402
                     parse_entries, parse_places)

ROOT = Path(__file__).resolve().parents[1]

# arm -> (Model keyword arguments, prompt level)
ARMS = {
    "nothink": (dict(max_tokens=1024), "plain"),
    "nothink_schema": (dict(max_tokens=1024, json_schema=SCHEMA), "plain"),
    "think256": (dict(max_tokens=4096, thinking=True, thinking_budget=256), "plain"),
    "think1k": (dict(max_tokens=4096, thinking=True, thinking_budget=1024), "plain"),
    "think4k": (dict(max_tokens=8192, thinking=True, thinking_budget=4096), "plain"),
    "think_unlimited": (dict(max_tokens=16384, thinking=True), "plain"),
    "nothink_corr": (dict(max_tokens=1024), "corr"),
    "think1k_corr": (dict(max_tokens=4096, thinking=True, thinking_budget=1024), "corr"),
    "think1k_corr_samp": (dict(max_tokens=4096, thinking=True, thinking_budget=1024), "samp"),
}
PROMPTS = {"plain": (SYSTEM, USER), "corr": (SYSTEM_CORR, USER_CORR), "samp": (SYSTEM_SAMP, USER_SAMP)}


def run_arm(client, model_name, name, arm, pause_s, limit):
    kwargs, level = ARMS[arm]
    model = Model(model_name, **kwargs)
    system, user = PROMPTS[level]
    out = ROOT / "preds" / f"{name}_{arm}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    texts = [json.loads(l) for l in open(ROOT / "data/eval_texts.jsonl")][:limit]
    for r in texts:
        if r["id"] in done:
            continue
        places, entries, c, tries = None, None, None, 0
        while places is None and tries < 2:
            tries += 1
            c = complete(client, model, system, user.format(text=r["text"]), f"{arm}/{r['id']}", pause_s)
            if level == "plain":
                places = parse_places(c.text)
            else:
                entries = parse_entries(c.text)
                places = None if entries is None else [t for t, *_ in entries]
        row = {"id": r["id"], "places": sorted({p.strip() for p in places or [] if p.strip()})}
        if level != "plain":
            row["corrected"] = sorted({(t.strip(), k.strip()) for t, k, _ in entries or []})
        if level == "samp":
            row["sampled"] = {t.strip(): s for t, _, s in entries or []}
        row.update(parse_failed=places is None, seconds=round(c.latency_s, 2), tries=tries,
                   output_tokens=c.output_tokens, reasoning_chars=c.reasoning_chars,
                   stop_reason=c.stop_reason, raw=c.text[-2000:])
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{arm}/{r['id']}: {row['places']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="model name as the server lists it (or $LLM_MODEL)")
    ap.add_argument("--name", help="prefix for the prediction files")
    ap.add_argument("--pause", type=float, default=float(os.environ.get("LLM_PAUSE") or 0),
                    help="seconds to wait after each answer (default: LLM_PAUSE or 0)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N texts")
    ap.add_argument("arms", nargs="+", choices=sorted(ARMS))
    a = ap.parse_args()
    if not a.model:
        ap.error("--model (or $LLM_MODEL) is required")
    client = make_client()
    for arm in a.arms:
        run_arm(client, a.model, a.name or a.model.rsplit("/", 1)[-1], arm, a.pause, a.limit)


if __name__ == "__main__":
    main()
