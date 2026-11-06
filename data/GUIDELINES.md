# Annotation guidelines: geographic place names

Task: list every geographic place name in the text, copied exactly as written (same spelling, case, and extent).

Include:
- countries, states, provinces, counties, cities, towns, villages, districts ("China", "east of china" -> "china")
- natural features: oceans, seas, gyres, currents, straits, bays, fjords, gulfs, rivers, lakes, mountains/ranges, deserts, islands, peninsulas, glaciers, plateaus, basins, trenches, ridges, hydrothermal vent fields, reefs ("South Pacific Gyre", "Disko Bay", "Mariana Trench")
- large regions and polar areas: "Arctic", "Antarctica", "Arctic Ocean", "Southern Ocean", "Tibetan Plateau", "Scandinavia", "Sub-Saharan Africa"
- direction words that are part of a conventional name: "West Greenland", "South China Sea", "Western Australia", "North Atlantic"
- named geographic sampling sites: parks, reserves, named lakes/caves/mines, named field stations and ocean time-series stations when they denote a place ("Station ALOHA", "Cedar Creek Ecosystem Science Reserve", "Yellowstone National Park")
- adjectival use when it denotes a place: "Arctic regions" -> "Arctic"; "Antarctic soils" -> "Antarctic"; "Mediterranean" (the sea/region)

Exclude:
- ships and cruise names ("RS Sonne", "SO245", "Ultrapac", "Tara Oceans" expedition name)
- institutions, companies, hospitals, universities, even when they contain a place name ("University of Copenhagen" -> nothing, but if the text separately says "Copenhagen", include that)
- organism names ("Atlantic salmon", "Alexandrium fundyense", "Bacillus subtilis subsp. ... " strain names)
- nationalities/ethnic or language adjectives used for people ("Chinese patients", "Japanese cohort") — do NOT include
- generic unnamed places ("the fjord", "11 stations", "coastal waters", "the operating room")
- sample IDs, project names, consortia, programmes ("Human Microbiome Project", "Earth Microbiome Project")
- cardinal directions alone ("north", "the south")

Extent: take the longest conventional name; don't split "South Pacific Gyre" into "South Pacific" + "Gyre". For "Uummannaq Fjord, Vaigat Strait and Disko Bay" give three names. For "Lake Biwa, Japan" give "Lake Biwa" and "Japan". Repeated mentions: list each distinct string once.

Output: a JSON list of distinct strings per text.
- 2026-10-28 gold edits after error review: added OK, AK (ERP009032), Luzon (SRP171601)
