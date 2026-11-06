"""Results table for every complete prediction file, with 95% bootstrap intervals (texts resampled).

usage: summarize.py > results/results.tsv
"""
import glob, json, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, load, score

gold = load(ROOT / "data/gold.jsonl")
ids = list(gold)
random.seed(0)
boots = [[random.choice(ids) for _ in ids] for _ in range(1000)]

print("system\tn_texts\texact_P\texact_R\texact_F1\tlenient_P\tlenient_R\tlenient_F1\tlenient_F1_lo\tlenient_F1_hi\ts_per_text\treasoning_chars_mean")
for path in sorted(glob.glob(str(ROOT / "preds/*.jsonl"))):
    preds = load(path)
    if len(preds) < len(ids):
        continue
    ex, _ = score(gold, preds, ids, False)
    ln, _ = score(gold, preds, ids, True)
    f1s = sorted(score(gold, preds, b, True)[0]["F1"] for b in boots)
    secs = statistics.mean(r.get("seconds", 0) for r in preds.values())
    rc = statistics.mean(r.get("reasoning_chars", 0) or 0 for r in preds.values())
    print("\t".join([Path(path).stem, str(len(ids))] + [f"{x:.3f}" for x in
          (ex["P"], ex["R"], ex["F1"], ln["P"], ln["R"], ln["F1"], f1s[25], f1s[974], secs)] + [f"{rc:.0f}"]))
