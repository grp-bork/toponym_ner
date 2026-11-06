"""Score place-name predictions against the gold standard.

Each text's gold and predicted places are sets of distinct strings. Two metrics:
  exact    the strings are identical after trimming punctuation and a leading "the"
  lenient  one contains the other, ignoring case ("Pacific Ocean" ~ "northeast Pacific Ocean")
Matching is one-to-one, so a single prediction cannot cover two gold places.
Gold "optional" strings are borderline cases (adjectives such as "Swiss soils", the annotators'
"unsure" lists): a prediction matching one is neither a hit nor a false positive.

usage: score.py [--split test|dev|all] [--errors] preds/*.jsonl
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRIP = " \t\n.,;:()[]\"'"


def norm(s):
    s = s.replace("\u2019", "'").replace("\u00b4", "'").replace("\u2018", "'").strip(STRIP)
    return s[4:] if s.lower().startswith("the ") else s


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def match(gold, pred, lenient):
    """Greedy one-to-one matching; returns matched pairs."""
    def ok(g, p):
        if g == p:
            return True
        if not lenient:
            return False
        g, p = g.lower(), p.lower()
        return g in p or p in g

    pairs, used = [], set()
    # exact matches first, so a lenient match cannot steal an exact one
    for g in sorted(gold, key=lambda g: g not in pred):
        for p in sorted(pred, key=lambda p: p != g):
            if p not in used and ok(g, p):
                pairs.append((g, p)); used.add(p); break
    return pairs


def score(gold, preds, ids, lenient):
    tp = fp = fn = 0
    errs = []
    for i in ids:
        g = {norm(s) for s in gold[i]["places"]} - {""}
        p = {norm(s) for s in preds.get(i, {}).get("places", [])} - {""}
        pairs = match(g, p, lenient)
        mg = {a for a, _ in pairs}; mp = {b for _, b in pairs}
        opt = {norm(o) for o in gold[i].get("optional", [])}
        mp |= {b for _, b in match(opt, p - mp, lenient)}
        tp += len(pairs); fp += len(p - mp); fn += len(g) - len(pairs)
        if g - mg or p - mp:
            errs.append((i, sorted(g - mg), sorted(p - mp)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(P=prec, R=rec, F1=f1, tp=tp, fp=fp, fn=fn), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--tsv", help="append results to this TSV")
    a = ap.parse_args()
    gold = load(ROOT / "data/gold.jsonl")
    ids = [i for i, r in gold.items() if a.split == "all" or r["split"] == a.split]
    print(f"{'system':55s} {'exP':>5s} {'exR':>5s} {'exF1':>5s} | {'lnP':>5s} {'lnR':>5s} {'lnF1':>5s}  n={len(ids)}")
    rows = []
    for path in a.preds:
        preds = load(path)
        missing = [i for i in ids if i not in preds]
        ex, _ = score(gold, preds, ids, False)
        ln, errs = score(gold, preds, ids, True)
        name = Path(path).stem
        print(f"{name:55s} {ex['P']:.3f} {ex['R']:.3f} {ex['F1']:.3f} | {ln['P']:.3f} {ln['R']:.3f} {ln['F1']:.3f}"
              + (f"  MISSING {len(missing)}" if missing else ""))
        rows.append([name, a.split, len(ids), len(missing)] + [f"{x[k]:.4f}" for x in (ex, ln) for k in ("P", "R", "F1")])
        if a.errors:
            for i, fn_, fp_ in errs:
                print(f"    {i}: missed={fn_} spurious={fp_}")
    if a.tsv:
        new = not Path(a.tsv).exists()
        with open(a.tsv, "a") as f:
            if new:
                f.write("system\tsplit\tn\tmissing\texact_P\texact_R\texact_F1\tlenient_P\tlenient_R\tlenient_F1\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()
