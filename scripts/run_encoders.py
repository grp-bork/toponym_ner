"""Run encoder NER models over the eval texts and write place-name predictions.

Every system writes preds/<name>.jsonl, one {"id", "places", "seconds"} per text.
Texts are cut into chunks of whole sentences, short enough for a 384/512-token encoder.

usage: run_encoders.py SYSTEM [SYSTEM ...]   (see SYSTEMS below; "all" runs every one)
"""
import json, os, re, sys, time
from pathlib import Path

import torch

torch.set_num_threads(min(32, os.cpu_count() or 1))
ROOT = Path(__file__).resolve().parents[1]
TEXTS = [json.loads(l) for l in open(ROOT / "data/eval_texts.jsonl")]
PREDS = ROOT / "preds"
PREDS.mkdir(exist_ok=True)


def chunks(text, max_words=180):
    """(offset, chunk) pairs of whole sentences, at most max_words words each."""
    sents = [(m.start(), m.group()) for m in re.finditer(r"[^\n.!?]+(?:[.!?]+(?=\s|$)|$)|\n+", text)]
    out, start, buf = [], None, ""
    for off, s in sents:
        if buf and len((buf + s).split()) > max_words:
            out.append((start, buf)); start, buf = None, ""
        if start is None:
            start = off
        buf = text[start:off + len(s)]
    if buf.strip():
        out.append((start, buf))
    return out


# --- GLiNER-style span models -------------------------------------------------

def gliner_system(model_id, labels, threshold):
    def run():
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_id)
        model.eval()

        def predict(text):
            found = []
            for off, c in chunks(text):
                for e in model.predict_entities(c, labels, threshold=threshold, flat_ner=True):
                    found.append(text[off + e["start"]: off + e["end"]])
            return found
        return predict
    return run


def gliner2_system(model_id, labels, threshold):
    def run():
        from gliner2 import GLiNER2
        model = GLiNER2.from_pretrained(model_id)

        def predict(text):
            found = []
            for off, c in chunks(text):
                r = model.extract_entities(c, labels, threshold=threshold, include_spans=True)
                for lab, ents in r.get("entities", {}).items():
                    for e in ents:
                        if isinstance(e, dict) and "start" in e:
                            found.append(text[off + e["start"]: off + e["end"]])
                        else:
                            found.append(e["text"] if isinstance(e, dict) else e)
            return found
        return predict
    return run


# --- classic token classifiers ------------------------------------------------

def hf_system(model_id, keep):
    """keep: entity groups (after B-/I- stripping) that count as places."""
    def run():
        from transformers import pipeline
        nlp = pipeline("token-classification", model=model_id, aggregation_strategy="first", device=-1)

        def predict(text):
            found = []
            for off, c in chunks(text, 150):
                for e in nlp(c):
                    if e["entity_group"].upper() in keep:
                        found.append(text[off + e["start"]: off + e["end"]])
            return found
        return predict
    return run


def spacy_system(model_name, keep):
    def run():
        import spacy
        nlp = spacy.load(model_name)
        return lambda text: [e.text for e in nlp(text).ents if e.label_ in keep]
    return run


LOC = ["location"]
GEO = ["location", "geographic feature", "body of water", "country", "city", "region"]
SYSTEMS = {
    # zero-shot span models
    "gliner_large-v2.1_loc": gliner_system("urchade/gliner_large-v2.1", LOC, 0.5),
    "gliner_large-v2.1_geo": gliner_system("urchade/gliner_large-v2.1", GEO, 0.5),
    "gliner_large-v2.1_geo_t0.3": gliner_system("urchade/gliner_large-v2.1", GEO, 0.3),
    "gliner_medium-v2.1_geo": gliner_system("urchade/gliner_medium-v2.1", GEO, 0.5),
    "gliner_multi-v2.1_geo": gliner_system("urchade/gliner_multi-v2.1", GEO, 0.5),
    "gliner_news-v2.1_geo": gliner_system("EmergentMethods/gliner_medium_news-v2.1", GEO, 0.5),
    "nuner_zero_geo": gliner_system("numind/NuNER_Zero", GEO, 0.5),
    "nuner_zero-span_geo": gliner_system("numind/NuNER_Zero-span", GEO, 0.5),
    "gliner2-large_geo": gliner2_system("fastino/gliner2-large-v1", GEO, 0.5),
    "gliner2.5-base_geo": gliner2_system("fastino/gliner2.5-base-v1", GEO, 0.5),
    # supervised NER, CoNLL/OntoNotes place classes
    "bert-large-conll_LOC": hf_system("dslim/bert-large-NER", {"LOC"}),
    "roberta-large-conll_LOC": hf_system("Jean-Baptiste/roberta-large-ner-english", {"LOC"}),
    "xlmr-large-conll_LOC": hf_system("FacebookAI/xlm-roberta-large-finetuned-conll03-english", {"LOC"}),
    "roberta-large-ontonotes_GPE-LOC": hf_system("tner/roberta-large-ontonotes5", {"GPE", "LOC"}),
    "spacy-trf_GPE-LOC": spacy_system("en_core_web_trf", {"GPE", "LOC"}),
    "spacy-trf_GPE-LOC-FAC": spacy_system("en_core_web_trf", {"GPE", "LOC", "FAC"}),
    # toponym-specific
    "geolm-toponym": hf_system("knowledge-computing/geolm-base-toponym-recognition", {"TOPO"}),
    "toponym-19thC-en": hf_system("Livingwithmachines/toponym-19thC-en", {"LOC", "BUILDING", "STREET", "TOPONYM"}),
    "reddit-place_names": hf_system("cjber/reddit-ner-place_names", {"LOCATION"}),
    # second round: more CoNLL-style and GLiNER variants
    "bert-base-conll_LOC": hf_system("dslim/bert-base-NER", {"LOC"}),
    "wikineural-multi_LOC": hf_system("Babelscape/wikineural-multilingual-ner", {"LOC"}),
    "deberta-v3-large-conll_LOC": hf_system("Gladiator/microsoft-deberta-v3-large_ner_conll2003", {"LOC"}),
    "dbmdz-bert-large-conll_LOC": hf_system("dbmdz/bert-large-cased-finetuned-conll03-english", {"LOC"}),
    "deberta-v3-large-fewnerd_location": hf_system("eek/deberta-v3-large-fewnerd-fine-ner", {"GPE", "BODIESOFWATER", "ISLAND", "MOUNTAIN", "OTHER", "PARK"}),
    "nuner-v2-fewnerd_location": hf_system("guishe/nuner-v2_fewnerd_fine_super", {"LOCATION_GPE", "LOCATION_BODIESOFWATER", "LOCATION_ISLAND", "LOCATION_MOUNTAIN", "LOCATION_OTHER", "LOCATION_PARK"}),
    "spacy-lg_GPE-LOC": spacy_system("en_core_web_lg", {"GPE", "LOC"}),
    "gliner_large-v2.5_geo": gliner_system("gliner-community/gliner_large-v2.5", GEO, 0.5),
    "gliner_large_news-v2.1_geo": gliner_system("EmergentMethods/gliner_large_news-v2.1", GEO, 0.5),
    "gliner_multitask-large-v0.5_geo": gliner_system("knowledgator/gliner-multitask-large-v0.5", GEO, 0.5),
    "gliner_large-v2.1_place_t0.7": gliner_system("urchade/gliner_large-v2.1", ["geographic place name"], 0.7),
    "gliner_news-v2.1_loc_t0.6": gliner_system("EmergentMethods/gliner_medium_news-v2.1", LOC, 0.6),
}


def main(names):
    if names == ["all"]:
        names = list(SYSTEMS)
    for name in names:
        out = PREDS / f"{name}.jsonl"
        if out.exists():
            print("skip", name); continue
        try:
            t0 = time.time()
            predict = SYSTEMS[name]()
            load_s = time.time() - t0
            rows = []
            for r in TEXTS:
                t = time.time()
                places = sorted(set(p.strip() for p in predict(r["text"]) if p.strip()))
                rows.append({"id": r["id"], "places": places, "seconds": round(time.time() - t, 3)})
            out.write_text("".join(json.dumps(x) + "\n" for x in rows))
            print(f"{name}: load {load_s:.0f}s, infer {sum(x['seconds'] for x in rows):.0f}s", flush=True)
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
