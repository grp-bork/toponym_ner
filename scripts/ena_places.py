"""List the geographic place names in ENA studies' titles and descriptions, with an LLM.

Input: project or study accessions (PRJEB/PRJNA/PRJDB or ERP/SRP/DRP), as arguments or one per line
in a file (-f). Output: a long TSV, one row per place:

  study_accession  secondary_study_accession  place  corrected  sampled

"place" is the name as written in the text, "corrected" its standard spelling (typos, capitals,
adjectives turned into the place: "Antartic" -> "Antarctic", "Alaskan" -> "Alaska"). "sampled" is
true if the study's samples come from the place or from somewhere inside it, false if the text names
it for another reason (background, comparison, isolates from elsewhere, a conference venue, a plan),
empty if the model gave no flag. A study with no place name gets one row with the last three empty,
so every study answered shows up. Accessions ENA does not know are reported on stderr.

The text sent is the study title, then the study name where it adds something (not part of the
title and not an organism name like "soil metagenome"), then the description. The study alias is
left out: for ENA submissions it is "ena-STUDY-<submitting centre>-...", which names the lab, not
the samples.

The model is asked with the benchmark's best setting: 1k thinking tokens, spelling correction and
the sampled flag (REPORT.md). Set OPENAI_BASE_URL and OPENAI_API_KEY first, then e.g.

  python scripts/ena_places.py --email you@example.org --model Qwen/... -o places.tsv PRJNA546078 ERP021705

With -o the output is appended and studies already in it are skipped, so a rerun resumes.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ena import fetch_studies, read_accessions  # noqa: E402
from llm import Model, complete, make_client  # noqa: E402
from prompts import SYSTEM_SAMP, USER_SAMP, parse_entries  # noqa: E402

COLUMNS = ["study_accession", "secondary_study_accession", "place", "corrected", "sampled"]


def study_text(r):
    title, name, desc = (r.get(k, "").strip() for k in ("study_title", "study_name", "study_description"))
    parts = [title]
    lname = name.lower()
    if name and lname not in title.lower() and not lname.endswith("metagenome") and not lname.startswith("uncultured"):
        parts.append(name)
    if desc and desc != title:
        parts.append(desc)
    return "\n\n".join(p for p in parts if p)


def flag(v):
    """JSON true/false (or their spellings as strings) as "true"/"false"; anything else as ""."""
    s = str(v).strip().lower()
    return s if s in ("true", "false") else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("accessions", nargs="*")
    ap.add_argument("--email", default=os.environ.get("ENA_EMAIL"), required=not os.environ.get("ENA_EMAIL"),
                    help="your contact address, sent to ENA with each request (or ENA_EMAIL in .env)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="model name as the server lists it (or $LLM_MODEL)")
    ap.add_argument("--thinking-budget", type=int, default=1024, help="thinking tokens (default %(default)s)")
    ap.add_argument("--pause", type=float, default=float(os.environ.get("LLM_PAUSE") or 0),
                    help="seconds to wait after each answer (default: LLM_PAUSE or 0)")
    ap.add_argument("-f", "--file", help="file with one accession per line")
    ap.add_argument("-o", "--out", help="output TSV (appended to; studies already in it are skipped)")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model (or $LLM_MODEL) is required")

    accs = list(a.accessions) + (read_accessions(a.file) if a.file else [])
    accs = list(dict.fromkeys(x.upper() for x in accs))
    if not accs:
        ap.error("no accessions given")
    model = Model(a.model, max_tokens=a.thinking_budget + 3072, thinking=True, thinking_budget=a.thinking_budget)

    done = set()
    if a.out and Path(a.out).exists():
        done = {row["study_accession"] for row in csv.DictReader(open(a.out), delimiter="\t")}
    studies = fetch_studies(accs, a.email, tool="ena-places")
    # a project can carry several secondary accessions: "SRP007749;SRP009392;SRP009438"
    found = {s["study_accession"] for s in studies} | {x for s in studies for x in s["secondary_study_accession"].split(";")}
    for x in accs:
        if x not in found:
            print(f"not found in ENA: {x}", file=sys.stderr)

    new_file = not (a.out and Path(a.out).exists() and Path(a.out).stat().st_size)
    out = open(a.out, "a") if a.out else sys.stdout
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    if new_file:
        w.writerow(COLUMNS)
    client = make_client()
    seen = set()
    for s in studies:
        acc = s["study_accession"]
        if acc in done or acc in seen:
            continue
        seen.add(acc)
        entries = None
        for _ in range(2):
            c = complete(client, model, SYSTEM_SAMP, USER_SAMP.format(text=study_text(s)), acc, a.pause)
            entries = parse_entries(c.text)
            if entries is not None:
                break
        if entries is None:
            print(f"no parsable answer for {acc}; left out, rerun to retry", file=sys.stderr)
            continue
        rows = sorted({(t.strip(), k.strip(), flag(v)) for t, k, v in entries if t.strip()})
        ids = [acc, s.get("secondary_study_accession", "")]
        for row in rows or [("", "", "")]:
            w.writerow(ids + list(row))
        out.flush()


if __name__ == "__main__":
    main()
