"""Fetch study records from the ENA Portal API.

ENA asks automated clients to identify themselves, so every request carries a contact e-mail
address, given by the user (--email or ENA_EMAIL in .env), in the User-Agent and Referer headers.
"""
import csv
import io
import re
import sys
import time
import urllib.parse
import urllib.request

import env  # noqa: F401  (loads .env)

ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
FIELDS = "study_accession,secondary_study_accession,study_title,study_name,study_description"


def check_email(email):
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""):
        raise SystemExit(f"--email must be a contact e-mail address, not {email!r}")
    return email


def headers(email, tool):
    return {"User-Agent": f"{tool}/0.1 (mailto:{check_email(email)})", "Referer": f"mailto:{email}"}


def fetch_studies(accs, email, tool="toponym-ner"):
    """ENA study records for project or study accessions (PRJ*, or ERP/SRP/DRP), 50 per request, 2 requests/s at most."""
    hdrs = headers(email, tool)
    out = []
    for i in range(0, len(accs), 50):
        batch = accs[i:i + 50]
        q = " OR ".join(f'{"study_accession" if a.startswith("PRJ") else "secondary_study_accession"}="{a}"'
                        for a in batch)
        data = urllib.parse.urlencode({"result": "study", "query": q, "fields": FIELDS,
                                       "format": "tsv", "limit": 0}).encode()
        req = urllib.request.Request(ENA_URL, data=data, headers=hdrs)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out += csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t")
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"ENA retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(2 ** attempt * 2)
        time.sleep(0.5)
    return out


def read_accessions(path):
    """One accession per line; blank lines and #-comments are skipped."""
    return [l.split("#")[0].strip() for l in open(path) if l.split("#")[0].strip()]
