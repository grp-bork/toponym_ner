"""The prompts used in the benchmark and by ena_places.py, and the parsers for their answers.

Three prompt levels, each extending the one before:
  SYSTEM / USER            place names only            -> {"places": ["...", ...]}
  SYSTEM_CORR / USER_CORR  plus a corrected spelling   -> {"places": [{"text", "corrected"}, ...]}
  SYSTEM_SAMP / USER_SAMP  plus a "sampled" flag       -> {"places": [{"text", "corrected", "sampled"}, ...]}
"""
import json
import re

SCHEMA = {"type": "object", "properties": {"places": {"type": "array", "items": {"type": "string"}}},
          "required": ["places"], "additionalProperties": False}


SYSTEM = """You extract geographic place names from scientific study titles and abstracts (ENA/SRA sequencing studies).

List every geographic place name, copied exactly as written in the text (same spelling, case and extent).

Include: countries, states, provinces, counties, cities, towns, districts; natural features (oceans, seas, gyres, currents, straits, bays, fjords, gulfs, rivers, lakes, mountains and ranges, deserts, islands, peninsulas, glaciers, plateaus, basins, trenches, seamounts, hydrothermal vent fields, reefs); large regions and polar areas ("Arctic", "Antarctica", "Tibetan Plateau", "Scandinavia"); direction words that are part of a conventional name ("West Greenland", "South China Sea"); named sampling sites that denote a place (national parks, reserves, named caves or mines, "Station ALOHA"); adjectives that denote a place ("Arctic regions" -> "Arctic", "Antarctic soils" -> "Antarctic").

Exclude: ships and cruise or expedition names ("RS Sonne", "SO245", "Ultrapac", "Tara Oceans"); institutions, universities, hospitals and companies, even if they contain a place name; organism and strain names ("Atlantic salmon", "Alexandrium fundyense"); nationality adjectives for people ("Chinese patients"); generic unnamed places ("the fjord", "coastal waters", "11 stations"); project and consortium names; bare compass directions.

Use the longest conventional name ("South Pacific Gyre", not "South Pacific"). List each distinct string once. If there are none, give an empty list."""

USER = 'Text:\n"""\n{text}\n"""\n\nAnswer with only a JSON object: {{"places": [...]}}'

SYSTEM_CORR = SYSTEM + """

For each place also give its corrected name: the standard English spelling and capitalisation, as a gazetteer would list it. Fix typos, missing or wrong capitals, stray plurals and odd punctuation ("Antartic" -> "Antarctic", "Atlantic Oceans" -> "Atlantic Ocean", "Canterbury basin" -> "Canterbury Basin", "Loki´s Castle" -> "Loki's Castle"). Keep the extent and the language of the name, and keep abbreviations as they are. If the name is already correct, repeat it unchanged."""

USER_CORR = ('Text:\n"""\n{text}\n"""\n\nAnswer with only a JSON object: '
             '{{"places": [{{"text": "<as written>", "corrected": "<corrected name>"}}, ...]}}')


SYSTEM_SAMP = SYSTEM_CORR + """

Also say for each place whether this study's samples come from it: true if samples were collected there or the place contains a sampling site (the country of a sampled lake, "Scandinavia" for samples from a Norwegian fjord); false if it is only mentioned for another reason: background ("a pathogen widespread in Africa"), comparison with other places or earlier studies, the origin of reference strains or clinical isolates studied elsewhere, a conference or institution, a plan or a negation. If the text doesn't say where the samples come from, use true for the places it names as the study's subject."""

USER_SAMP = ('Text:\n"""\n{text}\n"""\n\nAnswer with only a JSON object: '
             '{{"places": [{{"text": "<as written>", "corrected": "<corrected name>", "sampled": true}}, ...]}}')



def parse_places(text):
    """The list from the last {"places": [...]} object of strings in the reply, or None."""
    m = None
    for m in re.finditer(r"\{[^{}]*\"places\"\s*:\s*\[.*?\]\s*\}", text, re.S):
        pass
    if not m:
        return None
    try:
        return [p for p in json.loads(m.group())["places"] if isinstance(p, str)]
    except json.JSONDecodeError:
        return None


def parse_entries(text):
    """[(as written, corrected, sampled)] from the last {"places": [{...}, ...]} object in the reply, or None.

    corrected falls back to the text as written; sampled is whatever the model gave (None if absent)."""
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("places"), list):
            return [(p["text"], p.get("corrected") or p["text"], p.get("sampled")) for p in obj["places"]
                    if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return None
