"""Walk the whole PX-Web tree and save every table's metadata (IS + EN) to data/catalog.jsonl.

Resumable: tables already in the output file are skipped. Run:
    .venv/bin/python scripts/crawl_catalog.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagstofa import px  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "catalog.jsonl"
px.MIN_INTERVAL = 1.0  # be gentle with Hagstofa's public server: 1 request/second


def walk(path="", crumbs=()):
    """Yield (table_path, breadcrumb_texts, node) for every table under path."""
    for node in px.list_node(path):
        if "dbid" in node:  # root level lists databases: {dbid, text}
            node = {"id": node["dbid"], "type": "l", "text": node["text"]}
        child = f"{path}/{node['id']}".lstrip("/")
        if node["type"] == "l":
            yield from walk(child, crumbs + (node["text"],))
        elif node["type"] == "t":
            yield child, crumbs, node


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["path"] for line in OUT.open(encoding="utf-8")}
    n = 0
    with OUT.open("a", encoding="utf-8") as f:
        for path, crumbs, node in walk():
            if path in done:
                continue
            try:
                meta_is = px.metadata(path, "is")
            except Exception as e:  # noqa: BLE001 - log and keep crawling
                print(f"SKIP {path}: {e}", file=sys.stderr)
                continue
            try:
                meta_en = px.metadata(path, "en")
            except Exception:  # English version doesn't exist for every table
                meta_en = None
            rec = {
                "path": path,
                "id": node["id"].removesuffix(".px"),
                "updated": node.get("updated"),
                "breadcrumbs_is": list(crumbs),
                "title_is": meta_is.get("title") or node["text"].strip(),  # some tables lack a title
                "title_en": meta_en.get("title") if meta_en else None,
                "variables_is": meta_is["variables"],
                "variables_en": meta_en["variables"] if meta_en else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n} tables ({path})", flush=True)
    print(f"done, {n} new tables")


if __name__ == "__main__":
    main()
