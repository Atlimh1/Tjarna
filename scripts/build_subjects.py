"""Build data/subjects.jsonl: Statistics Iceland's subject folders (depth 3 of the px tree, ~135) with the everyday
words people use for them, written once by the model. Used only after a "no data" answer, to find tables on the
question's subject that the search missed (suggest.py): "hvað eru framleidd mörg epli á íslandi?" found GDP and
emissions tables, while LAN10103 "Uppskera og afurðir" sits in Landbúnaður > Búfé og uppskera.

    LLM_BASE_URL=... LLM_MODEL=gemma LLM_API_KEY=... .venv/bin/python scripts/build_subjects.py

Resumable: folders already in the file are skipped, each new one is written as soon as it is done. The file is meant
to be read and corrected by hand (one folder per line); a rerun keeps the corrections.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagstofa import llm  # noqa: E402
from hagstofa.retrieve import DATA, load_catalog  # noqa: E402

OUT = DATA / "subjects.jsonl"
DEPTH = 3
TITLES_SHOWN = 15
SYSTEM = (
    "You help a search engine for Statistics Iceland's tables. Given one subject folder (its name and the titles "
    "of its tables), list the everyday words people use when asking about this subject: specific nouns for the "
    "things counted or measured (e.g. for crops: epli, kartöflur, grænmeti, korn, hey, uppskera). 10-25 words in "
    "Icelandic (base form) and 10-25 in English. NOT generic words that fit any subject (fjöldi, tölur, ár, Ísland, "
    "framleiðsla, hagtölur, number, total, production, Iceland). "
    'Answer only with JSON: {"is": ["..."], "en": ["..."]}'
)


def folders():
    """{folder path: {"name": "A › B › C", "tables": [(id, title)]}} at DEPTH (shallower tables: their own folder)."""
    out = defaultdict(lambda: {"name": None, "tables": []})
    for t in load_catalog():
        parts = t["path"].split("/")[:-1]  # without the .px file
        key = "/".join(parts[:DEPTH])
        out[key]["name"] = out[key]["name"] or " › ".join(t["breadcrumbs_is"][:DEPTH])
        out[key]["tables"].append((t["id"], t["title_is"]))
    return out


def words(name, tables):
    titles = "\n".join(f"- {title}" for _, title in tables[:TITLES_SHOWN])
    reply = llm.parse_json(llm.chat(SYSTEM, f"Folder: {name}\nTables:\n{titles}", json_mode=True))
    clean = lambda ws: list(dict.fromkeys(w.strip().lower() for w in ws if isinstance(w, str) and w.strip()))  # noqa: E731
    return clean(reply.get("is") or []), clean(reply.get("en") or [])


def main():
    done = {json.loads(line)["folder"] for line in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    todo = {k: v for k, v in sorted(folders().items()) if k not in done}
    print(f"{len(done)} folders done, {len(todo)} to go")
    with open(OUT, "a", encoding="utf-8") as f:
        for i, (key, folder) in enumerate(todo.items(), 1):
            try:
                words_is, words_en = words(folder["name"], folder["tables"])
            except Exception as e:  # noqa: BLE001 - a cut-off or broken reply: left for the next run
                print(f"{i}/{len(todo)} {folder['name']}: skipped ({type(e).__name__}), retried on the next run", flush=True)
                continue
            f.write(json.dumps({"folder": key, "name": folder["name"], "words_is": words_is, "words_en": words_en,
                                "tables": [tid for tid, _ in folder["tables"]]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {folder['name']}: {', '.join(words_is[:8])}", flush=True)


if __name__ == "__main__":
    main()
