"""Compare Qwen's per-place "sampled" flag with the hand-checked labels in data/sampled_labels/labels.jsonl.

A predicted place is matched to a labelled gold string as in score.py's lenient metric
(case-insensitive containment). "unclear" labels are reported but not scored.

usage: score_sampled.py preds/qwen3.8_think1k_corr_samp.jsonl
"""
import collections, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
labels = collections.defaultdict(dict)
for path in glob.glob(str(ROOT / "data/sampled_labels/labels.jsonl")):
    for r in map(json.loads, open(path)):
        labels[r["id"]][r["place"]] = r["label"]

conf = collections.Counter()
listing = []
for r in map(json.loads, open(sys.argv[1])):
    for place, flag in r.get("sampled", {}).items():
        gold = next((lab for g, lab in labels[r["id"]].items()
                     if g.lower() == place.lower()), None) or next(
            (lab for g, lab in labels[r["id"]].items() if g.lower() in place.lower() or place.lower() in g.lower()), None)
        if gold is None:
            gold = "unlabelled"
        conf[gold, flag] += 1
        if gold != "sampled" or flag is not True:
            listing.append((r["id"], place, gold, flag))

print("label        flag=true  flag=false  other")
for lab in ("sampled", "not_sampled", "unclear", "unlabelled"):
    print(f"{lab:12s} {conf[lab, True]:9d}  {conf[lab, False]:10d}  {sum(v for (l, f), v in conf.items() if l == lab and f not in (True, False)):5d}")
print()
for i, p, lab, flag in sorted(listing):
    print(f"{i}\t{p}\tlabel={lab}\tflag={flag}")
