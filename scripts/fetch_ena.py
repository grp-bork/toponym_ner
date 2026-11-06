"""Fetch ENA titles, names and descriptions for a list of study accessions, as JSON.

usage: fetch_ena.py --email you@example.org data/source_studies.txt ena_studies.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ena import fetch_studies, read_accessions  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--email", default=os.environ.get("ENA_EMAIL"), required=not os.environ.get("ENA_EMAIL"),
                help="your contact address, sent to ENA with each request (or ENA_EMAIL in .env)")
ap.add_argument("accessions", help="file with one accession per line")
ap.add_argument("out", help="output JSON")
a = ap.parse_args()

accs = read_accessions(a.accessions)
rows = fetch_studies(accs, a.email, tool="toponym-ner-fetch")
json.dump(rows, open(a.out, "w"), indent=1)
print(len(accs), "requested,", len(rows), "returned", file=sys.stderr)
