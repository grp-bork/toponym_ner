# Finding place names in ENA study titles and abstracts: model comparison

28–29 October 2026

## Question

Which model best lists the geographic place names in an ENA study's title and description, so they can be passed to a geocoder? The task covers detection only, with one label for all places; natural features count as places, and ships and cruises don't. No fine-tuning.

## Data

- **Studies:** the 1,099 studies flagged in the MicrobeAtlas coordinate audit. ENA returned a title and description for 1,098 of them.
- **Evaluation set:** 200 of these studies, sampled at random (seed 42), plus two hand-picked example texts (`USER1`, `USER2`). Each text is the title, a blank line, then the description.
- **Gold standard:** two independent annotators per text (Sonnet agents, following `data/GUIDELINES.md`). They agreed on 180 place strings, and 13 disagreements were settled by hand. Three places were added after reviewing model errors.
  - The result is 191 place mentions (148 distinct names).
  - Another 37 borderline strings, such as "Swiss soils" and bare "Cayman", are marked optional: predicting one is neither rewarded nor penalised.
- **Scoring:**
  - *Lenient* (the main metric): a prediction matches a gold place if one contains the other, ignoring case, so "northern Gulf of Mexico" matches "Gulf of Mexico".
  - *Exact*: the strings must be the same, apart from surrounding punctuation and a leading "the".
  - Matching is one-to-one.
  - The 95% intervals come from 1,000 bootstrap resamples of texts.

## Systems tested

- **Encoders, CPU only** (no GPU on this host):
  - 17 supervised NER models, keeping their place classes (CoNLL LOC, OntoNotes GPE/LOC/FAC, Few-NERD location types): BERT, RoBERTa, XLM-R and DeBERTa variants, spaCy (trf and lg), and three toponym-specific models (GeoLM, toponym-19thC, reddit place names).
  - 12 zero-shot span models: GLiNER v2.1/v2.5, news and multitask variants, NuNER Zero, and GLiNER2, with several label sets and thresholds.
  - Two ensembles of the best three encoders.
- **LLMs served with vLLM** behind an OpenAI-compatible endpoint (`scripts/run_llm.py`): one request at a time, a 2 s pause between requests, and backoff on errors.
  - **Qwen 3.8 27B** (NVFP4 quantisation): no thinking (three repeat runs, plus one with JSON-schema-constrained output), and thinking budgets of 256 tokens, 1k, 4k and unlimited.
  - **DeepSeek-V4-Flash:** default mode only.
- **Spelling correction:** the Qwen prompt was extended to also return a corrected spelling for each place, tested with no thinking and with 1k thinking.
- **Sampled flag:** the prompt was further extended to say, for each place, whether the study's samples come from it (1k thinking only). The flag was checked against hand-reviewed labels; see "Places that aren't sampling sites" below.

The Qwen prompt contains essentially the same guidelines the gold standard was annotated with, which favours the LLMs somewhat over the encoders.

## Results (n = 202 texts)

| System | Lenient F1 (95% CI) | Precision | Recall | Exact F1 | Time per text |
|---|---|---|---|---|---|
| **Qwen 3.8, 1k thinking** | **0.990** (0.978–0.998) | 0.990 | 0.990 | 0.927 | 2.3 s + 2 s pause |
| Qwen 3.8, unlimited thinking | 0.989 (0.977–1.000) | 1.000 | 0.979 | 0.936 | 3.6 s + 2 s |
| Qwen 3.8, 4k thinking | 0.987 | 0.984 | 0.990 | 0.935 | 3.7 s + 2 s |
| Qwen 3.8, 1k thinking, with spelling correction | 0.984 | 0.989 | 0.979 | 0.905 | 3.0 s + 2 s |
| Qwen 3.8, 1k thinking, with spelling correction and sampled flag | 0.979 | 0.984 | 0.974 | 0.891 | 3.6 s + 2 s |
| Qwen 3.8, 256 thinking | 0.976 | 0.984 | 0.969 | 0.907 | 1.3 s + 2 s |
| Qwen 3.8, no thinking (3 runs) | 0.968–0.971 | ~0.98 | ~0.955 | 0.883–0.910 | 0.45 s + 2 s |
| DeepSeek-V4-Flash, default | 0.965 (0.931–0.989) | 0.994 | 0.937 | 0.878 | 3.7 s + 2 s |
| Qwen 3.8, no thinking, with spelling correction | 0.954 | 0.978 | 0.932 | 0.876 | 0.57 s + 2 s |
| dslim/bert-large-NER (LOC), best encoder | 0.915 (0.874–0.947) | 0.960 | 0.874 | 0.837 | 0.16 s, CPU |
| 2-of-3 encoder vote | 0.914 | 0.976 | 0.859 | 0.852 | – |
| spaCy en_core_web_trf (GPE+LOC+FAC) | 0.909 | 0.902 | 0.916 | 0.828 | 0.08 s, CPU |
| reddit-ner-place_names | 0.868 | 0.913 | 0.827 | 0.720 | 0.08 s |
| other CoNLL/OntoNotes/Few-NERD encoders | 0.74–0.86 | | | | |
| GeoLM toponym, toponym-19thC | 0.76, 0.83 | | | | |
| GLiNER / NuNER Zero / GLiNER2 (zero-shot) | 0.59–0.78 | | | | |

Every system's numbers are in `results/results.tsv`.

## Findings

- **LLMs beat encoders clearly.** The best encoder misses about one place in eight, mostly in three cases:
  - lower-case or unusual names ("Bundera sinkhole", "Kam Kotia Mine")
  - natural features
  - places inside run-together tokens

  Zero-shot GLiNER-style models over-predict badly: generic phrases like "coastal waters" and organisations.
- **A little thinking helps; more doesn't.** Recall rises from 0.95 with no thinking to 0.99 with a 1k budget. Larger budgets cost time without gaining accuracy: unlimited thinking averaged 2,300 reasoning characters, and its longest answer used 24,600.
- **No thinking is the fast option.** It gives up about 2 points of F1 and is the least stable: three identical runs gave different outputs on 31 of 202 texts.
- **JSON-schema-constrained output didn't help** (0.968 against 0.971).
- **Qwen ran cleanly:** about 2,000 requests, with no parse failures and no server errors.
- **DeepSeek-V4-Flash** is no better than Qwen and is slower.
  - The deployment ignores the switch that turns thinking off. One answer used up its whole token limit on reasoning and came back empty.
  - The deployment we used rejected thinking budgets with an HTTP 500 ("reasoning_config is not configured"), so only the default mode was tested.
- **Spelling correction works**, at little cost to detection (1k thinking: 0.984 against 0.990, about run-to-run noise). All the misspellings in the gold set were fixed:
  - Antartic, Atlantic Oceans, Loki´s, Porto of Leixões, Qilian Mountain, Koean, canadian Archipelago
  - lower-case "basin", "sea", "seamount" and "strait"

  It also does more than the prompt asks:
  - it turns adjectives into places (Alaskan → Alaska)
  - it expands NZ and US, although the prompt says to keep abbreviations
  - it capitalises direction words ("Northern Gulf of Mexico")
  - occasionally it renames a place: Malagasy Highlands → Madagascar Highlands, and once, without thinking, Alatsee → "Lake Alat", which is wrong

  Four of the misspellings are also the examples in the prompt, so for those this test doesn't show it generalises.

## Places that aren't sampling sites

A text can name places where none of its samples were taken. Four Sonnet agents, plus a hand check of one batch and of every case they flagged, labelled all 228 gold and optional strings:

| | Gold places (191) | Optional strings (37) |
|---|---|---|
| Sampling site, or a region containing it | 180 | 25 |
| Not a sampling site | 6 (3%), in 5 of the 87 texts with places | 8 |
| Unclear | 5 | 4 |

None of these are literal negations ("not found in X"). They are:
- background ("an emerging pest in Europe", for a study sampled in Argentina)
- a conference venue ("Samples presented at EMP conference Shenzhen"; where the samples came from isn't stated)
- a planned Red Sea–Dead Sea conduit
- clinical isolates from South America, for a study of Venezuelan soils

Only about one text in 40 has a place that would point the geocoder somewhere wrong. Usually the background place is a large region that still contains the real site, which does no harm. The optional not-sampled strings are mostly adjectives and non-places: Earth, Mars, "Asian dust", "EU legislation", "San Joaquin Valley fever".

**The sampled flag**, compared with these labels (8 not-sampled strings matched a prediction):

| | Flagged sampled | Flagged not sampled |
|---|---|---|
| Labelled a sampling site (189) | 187 | 2 |
| Labelled not a sampling site (8) | 1 | 7 |
| Labelled unclear (5) | 5 | 0 |

- **It catches the misleading cases:** Shenzhen/China, "Europe" for the Argentinian pest, the Red Sea conduit.
- **Its disagreements are defensible:**
  - "Europe" in "the most polluted river in Europe" → sampled; Europe contains the site.
  - "Arctic" → not sampled; it appears only in background sentences, although it contains southwest Greenland.
  - "near Timmins" → not sampled; a reference town rather than the site.
- **Unclear places default to sampled**, as the prompt says.
- **Borderline flags are not stable between runs.** In a later run, "South America" for the Venezuela study came back sampled rather than not sampled; both are defensible.
- **Small sample:** with only 8 negatives, this shows the kinds of mistakes the flag makes, not a reliable error rate.
- **Cost:** the flag costs about half a point of lenient F1, roughly one more detection error in 202 texts, which is within run-to-run noise, plus about 0.6 s per text.
- **No leakage:** its prompt examples were written fresh, not taken from the evaluation texts.

## Which text fields to use

The evaluation used the title and description only. Across the audit studies, `study_name` adds a place name in only about 8 studies, e.g. "Dead Sea Springs", "Jan Mayen Vent Fields" and "Carajas lakes". `study_alias` should not be used: for ENA submissions it is `ena-STUDY-<submitting centre>-<date>`, which names the submitter's institution, not where the samples came from.

## Recommendation

Use **Qwen 3.8 with a 1k thinking budget, the spelling-correction prompt and the sampled flag**. It handles about 15,000 studies a day with one request at a time. When checking coordinates, compare samples against the places flagged as sampled, falling back to all places when a study has none. For all of ENA, or as a quick first pass, `dslim/bert-large-NER` on CPU is about 30 times faster, at 0.915 F1.

`scripts/ena_places.py` implements the recommendation. It takes project or study accessions, fetches the records from ENA, builds the text (title, then study name where it adds something, then description), and writes a long TSV with one row per place:

```
study_accession  secondary_study_accession  place  corrected  sampled
```

`sampled` is `true`, `false`, or empty if the model left it out. A study with no place gets one row with the last three columns empty.

## Files

- `data/`: evaluation texts, gold standard, annotation guidelines, the source study list
- `data/annotation/`: the two independent annotations and their disagreements
- `data/sampled_labels/`: per-place sampled / not-sampled labels, with quotes
- `results/results.tsv`: full results table
- `scripts/`: see the README
