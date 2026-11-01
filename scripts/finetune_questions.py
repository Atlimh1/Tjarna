"""Step 1 of the search-model fine-tuning: an LLM writes questions for tables that are in NO eval set.

The training pairs (question -> table) must never touch a table we measure on, or the eval would reward
memory instead of learning. So every table in dev, test, hand, muni, real and e2e is left out (~1800 of
~2100 tables remain). Archive tables are left out too: the app down-weights them anyway.

The prompt asks for the same mix as the held-out sets: specific, casual (everyday words, as people ask) and
variant questions, mostly Icelandic plus one English. Writes data/finetune/train_questions.jsonl, one line per
question {"q", "table", "lang", "style"}; resumable (tables already done are skipped), so it can be stopped
and restarted. Calls run in parallel threads; vLLM/Ollama batch them on the GPU.

    LLM_BASE_URL=... LLM_MODEL=... LLM_REASONING=none .venv/bin/python scripts/finetune_questions.py
    ... --limit 20          # try a few tables first, then read the questions before running all of them
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from hagstofa import llm  # noqa: E402
from hagstofa.retrieve import is_archive, load_catalog  # noqa: E402
from gagnvist_eval import starter_path  # noqa: E402
from run_eval import load_questions  # noqa: E402
from sample_tables import describe  # noqa: E402

OUT = ROOT / "data" / "finetune" / "train_questions.jsonl"
EVAL_SETS = ["dev", "test", "hand", *(str(p) for p in sorted((ROOT / "eval").glob("questions_*.jsonl"))),
             str(starter_path() / "questions" / "examples.jsonl")]  # the organizers' 60 questions

SYSTEM = (
    "Þú býrð til æfingagögn fyrir leitarvél yfir töflur Hagstofu Íslands. Þú færð lýsingu á einni töflu og skrifar "
    "spurningar sem venjulegt fólk gæti spurt og þessi tafla svarar. Fjölbreyttar: ólík orð, ekki afrita titilinn.\n"
    "- 2 'specific': nákvæmar spurningar á íslensku sem nefna gildi úr töflunni (t.d. ár, kyn, svæði).\n"
    "- 2 'casual': á hversdagslegu íslensku máli, eins og fólk talar (t.d. 'hvað búa margir...', 'eru fleiri...').\n"
    "- 1 'english': á ensku, eins og ferðamaður eða innflytjandi myndi spyrja.\n"
    'Svaraðu eingöngu með JSON: {"questions": [{"q": "...", "style": "specific|casual|english"}, ...]}'
)


def tables_of(item):
    """The tables a question is scored on: ours list them in "table"; the organizers' format (their 60 and
    questions_new) in each part's "accepted"; no-data questions have none."""
    if "table" in item:
        return item["table"] if isinstance(item["table"], list) else [item["table"]]
    parts = [p for p in item.get("parts") or [] if isinstance(p, dict)]  # questions_mixed: parts carry no table
    return [t for p in parts for t in (p.get("accepted") or ([p["table"]] if "table" in p else []))]


def held_out_tables():
    out = set()
    for name in EVAL_SETS:
        try:
            for item in load_questions(name):
                out.update(tables_of(item))
        except FileNotFoundError:
            pass
    for f in (ROOT / "eval" / "heldout").glob("tables_*.json"):  # sampled for dev/test, with or without questions
        out.update(t["id"] for t in json.loads(f.read_text(encoding="utf-8")))
    return out


def questions_for(table):
    reply = llm.chat_json(SYSTEM, json.dumps(describe(table), ensure_ascii=False, indent=1))
    out = []
    for item in reply.get("questions", []):
        q = (item.get("q") or "").strip() if isinstance(item, dict) else ""
        if len(q) >= 10:
            style = item.get("style") or ""
            out.append({"q": q, "table": table["id"], "lang": "en" if style == "english" else "is", "style": style})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only this many tables (try it first)")
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    args = ap.parse_args()

    held_out = held_out_tables()
    seen, tables = set(), []
    for t in load_catalog():
        if t["id"] not in held_out and t["id"] not in seen and not is_archive(t):  # some ids are in two folders
            seen.add(t["id"])
            tables.append(t)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["table"] for line in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    todo = [t for t in tables if t["id"] not in done][: args.limit]
    print(f"{len(held_out)} tables held out (eval sets); {len(tables)} trainable; {len(done)} done; {len(todo)} to do")

    lock, failed, written = threading.Lock(), 0, 0
    with open(OUT, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(questions_for, t): t for t in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001 - bad JSON from one table shouldn't stop the run; rerun retries it
                failed += 1
                print(f"  {futures[f]['id']}: {type(e).__name__}: {str(e)[:80]}")
                continue
            with lock:
                out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                out.flush()
                written += len(rows)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} tables, {written} questions, {failed} failed")
    print(f"done: {written} questions -> {OUT}" + (f"; {failed} tables failed, rerun to retry them" if failed else ""))


if __name__ == "__main__":
    main()
