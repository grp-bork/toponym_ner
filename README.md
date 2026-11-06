# Place names in ENA study descriptions

A small benchmark and a ready-to-use script for finding geographic place names (countries, cities, seas, lakes, mountains, vent fields, …) in the titles and descriptions of ENA/SRA sequencing studies. The goal is to give a geocoder the places a study's samples came from.

- **Benchmark:** 202 study texts with double-annotated, adjudicated gold place names (191 mentions), scored with a lenient (containment) and an exact metric.
- **Comparison:** about 30 encoder NER models (CoNLL/OntoNotes/Few-NERD, spaCy, GLiNER-style zero-shot models, toponym-specific models) and two open-weights LLMs.
- **Result:** Qwen 3.8 27B with a 1k-token thinking budget reaches a lenient F1 of 0.990. The best encoder, `dslim/bert-large-NER`, reaches 0.915. Details are in [REPORT.md](REPORT.md).
- **Production script:** `scripts/ena_places.py` takes ENA accessions and writes one row per place, with a corrected spelling and a flag saying whether the study's samples come from that place.

## Requirements

- Python 3.9 or later
- For the LLM scripts: an **OpenAI-compatible server** running an open-weights model. The benchmark used [vLLM](https://docs.vllm.ai/). Thinking is switched off with `chat_template_kwargs={"enable_thinking": false}` and capped with vLLM's `thinking_token_budget` request parameter. Servers that ignore these still run the scripts, but their results won't be comparable with the report. The report's LLM results come from Qwen 3.8 27B in NVFP4 quantisation, on a shared vLLM deployment.
- For the encoder baselines (optional): PyTorch, transformers, GLiNER, GLiNER2 and spaCy. They run on CPU.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # LLM scripts, ENA, scoring
.venv/bin/pip install -r requirements-encoders.txt   # optional: encoder baselines
```

## Configuration

Copy `.env.example` to `.env` (git ignores it) and fill it in. Variables already set in the shell take precedence.

```sh
OPENAI_BASE_URL=http://localhost:8000/v1   # your server
OPENAI_API_KEY=...                         # any value if the server has no key
# OPENAI_API_KEY_FILE=~/.config/llm_key    # or read the key from a file
LLM_MODEL=<model name as the server lists it>   # or pass --model
# LLM_PAUSE=2                              # seconds after each answer, for a shared server
ENA_EMAIL=you@example.org                  # or pass --email
```

For example, a vLLM server for a Qwen 3 model:

```sh
vllm serve <hf-model-id> --reasoning-parser qwen3
```

Without a reasoning parser, the thinking ends up in the answer text instead of a separate field.

Scripts that call ENA require a contact e-mail address, given by **`--email`** or `ENA_EMAIL`. It is sent in the `User-Agent` and `Referer` headers, as ENA asks of automated clients.

### Checking the setup

```sh
.venv/bin/python scripts/check_setup.py --model "$LLM_MODEL" --email you@example.org [--encoders]
```

The check confirms that the server is reachable and lists the model, that thinking can be switched off, that a small thinking budget is honoured, and that one benchmark text gives parsable JSON with the expected places. With `--email` it also queries ENA; with `--encoders` it imports the encoder packages. Each check prints `OK`, `WARN` or `FAIL`. A `WARN` means the scripts will run but the results may differ from the report.

## Finding places in ENA studies

```sh
.venv/bin/python scripts/ena_places.py --email you@example.org -o places.tsv PRJNA546078 ERP021705
.venv/bin/python scripts/ena_places.py --email you@example.org -o places.tsv -f accessions.txt
```

The script accepts project (`PRJEB…`, `PRJNA…`, `PRJDB…`) and study (`ERP…`, `SRP…`, `DRP…`) accessions. The output is a TSV with one row per place. Here is an excerpt from a run on five studies:

| study_accession | secondary_study_accession | place | corrected | sampled |
|---|---|---|---|---|
| PRJEB34393 | ERP117289 | Antartic | Antarctic | true |
| PRJEB34393 | ERP117289 | McMurdo Dry Valleys | McMurdo Dry Valleys | true |
| PRJEB33602 | ERP116408 | Porto of Leixões harbor | Port of Leixões Harbor | true |
| PRJNA453402 | SRP142538 | Argentina | Argentina | true |
| PRJNA453402 | SRP142538 | Europe | Europe | false |
| PRJEB2962 | ERP001347 | Dead Sea | Dead Sea | true |
| PRJEB2962 | ERP001347 | Red Sea | Red Sea | false |

The two `false` rows are places the text names without sampling there. SRP142538 calls the studied pest "an emerging pest in Europe", but the samples come from Argentina. ERP001347 mentions a planned Red Sea–Dead Sea water conduit.

- `place` is the name as written in the text, and `corrected` is its standard spelling.
- `sampled` is `false` when the text names the place only for background, comparison, the origin of reference strains, a conference venue and the like.
- A study with no place name gets one row with the last three columns empty.
- With `-o`, the file is appended to and studies already in it are skipped, so an interrupted run resumes.
- `--pause S` waits S seconds after each answer, for shared servers.
- `--thinking-budget` defaults to the benchmark's 1024 tokens.

## Reproducing the benchmark

Predictions go to `preds/`, which is not part of the repository.

```sh
# LLM arms: nothink, nothink_schema, think256, think1k, think4k, think_unlimited,
#           nothink_corr, think1k_corr, think1k_corr_samp
.venv/bin/python scripts/run_llm.py --model "$LLM_MODEL" --name qwen --pause 2 think1k think1k_corr_samp

# encoder baselines ("all", or names from SYSTEMS in the script)
.venv/bin/python scripts/run_encoders.py bert-large-conll_LOC spacy-trf_GPE-LOC-FAC

# scoring
.venv/bin/python scripts/score.py --split all preds/*.jsonl          # exact and lenient P/R/F1
.venv/bin/python scripts/summarize.py > results.tsv                 # all systems, with bootstrap CIs
.venv/bin/python scripts/score_sampled.py preds/qwen_think1k_corr_samp.jsonl   # the sampled flag
.venv/bin/python scripts/ensemble.py NAME K preds/a.jsonl preds/b.jsonl ...      # K-of-N vote
```

`run_llm.py` resumes where it stopped. For repeat runs of the same arm, give each run a different `--name`. The LLM runs in the report used a 2 s pause between requests; the per-text times in the report exclude that pause.

## Layout

```
REPORT.md                     method, results, findings
data/
  eval_texts.jsonl            202 texts: {id, split, text}; text = title, blank line, description
  gold.jsonl                  gold places per text: {id, split, places, optional}
  GUIDELINES.md               annotation guidelines
  annotation/                 the two independent annotations and their disagreements
  sampled_labels/labels.jsonl per-place sampled / not_sampled / unclear labels, with quoted evidence
  source_studies.txt          the 1,099 ENA studies the evaluation texts were drawn from
results/results.tsv           scores of every system in the report
scripts/
  check_setup.py              test the server, ENA and packages
  ena_places.py               production script: accessions -> places TSV
  run_llm.py, run_encoders.py run the benchmark systems
  score.py, summarize.py, score_sampled.py, ensemble.py
  fetch_ena.py                fetch ENA titles/descriptions for an accession list
  llm.py, ena.py, prompts.py  shared client, ENA access, prompts and answer parsers
```

## Data

- **Texts:** taken from public ENA study records (title and description). `USER1` and `USER2` are two hand-picked examples. To refetch the records, run `scripts/fetch_ena.py --email you@example.org data/source_studies.txt out.json`.
- **Gold standard:** two independent LLM annotators (Claude Sonnet) followed `data/GUIDELINES.md`. Their disagreements were settled by hand, and three places were added after reviewing model errors.
- **Optional strings:** 37 borderline strings are marked `optional`. Predicting one is neither rewarded nor penalised.
- **Split:** `split` divides the texts into `dev` and `test`; the report scores all 202.

## License

MIT; see [LICENSE](LICENSE). The study texts come from ENA and are covered by [EMBL-EBI's terms of use](https://www.ebi.ac.uk/about/terms-of-use).
