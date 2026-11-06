"""Combine encoder predictions: union, or votes of at least k systems (lenient string match).

usage: ensemble.py NAME K preds/a.jsonl preds/b.jsonl [...]
Writes preds/ens_NAME.jsonl. A place proposed by several systems in different extents
("Gulf of Mexico" vs "northern Gulf of Mexico") counts once, under its most common spelling.
"""
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(name, k, paths):
    systems = [{r["id"]: r for r in map(json.loads, open(p))} for p in paths]
    rows = []
    for i in systems[0]:
        votes = collections.Counter()
        for s in systems:
            for p in set(s[i]["places"]):
                votes[p] += 1
        # merge strings that contain each other, case-insensitively
        groups = []
        for p, _ in votes.most_common():
            for g in groups:
                if any(p.lower() in q.lower() or q.lower() in p.lower() for q in g):
                    g.append(p); break
            else:
                groups.append([p])
        keep = []
        for g in groups:
            support = len({j for j, s in enumerate(systems) if set(s[i]["places"]) & set(g)})
            if support >= k:
                keep.append(max(g, key=lambda p: (votes[p], len(p))))
        rows.append({"id": i, "places": sorted(keep),
                     "seconds": round(sum(s[i].get("seconds", 0) for s in systems), 3)})
    (ROOT / f"preds/ens_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
