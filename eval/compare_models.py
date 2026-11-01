"""Compare LLMs on the same questions, so the model is chosen by measurement, not reputation.

Three measurements per model (eval/questions_e2e.jsonl by default: questions with `truth`):
  pick    the LLM chooses among the retriever's top-k: right table?          (no Hagstofa API calls)
  oracle  the LLM gets the RIGHT table: right values chosen? right number in the answer?
          This is the LLM alone, without retrieval errors: the place to tune prompts.
  full    the whole pipeline as the app runs it: right number in the answer?
oracle and full query the PX-Web API: one query per answer, plus the expected numbers, which are kept in
eval/results/expected.json so later runs don't fetch them again. Every answer is saved to
eval/results/<model>[_reasoning-X]_<time>.jsonl, so you can read and judge the Icelandic.

    LLM_BASE_URL=http://<gpu-machine>:11434/v1 .venv/bin/python eval/compare_models.py gemma4:26b gpt-oss:20b
    ... compare_models.py gemma4:e2b --modes pick                 # quick, no API calls
    ... compare_models.py gemma4:26b --modes oracle -n 5           # first 5 questions only
    ... compare_models.py gemma4:26b --reasoning none              # thinking off (same as LLM_REASONING=none)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from run_eval import accepted, answer_ok, expected_numbers, expected_value, has_answer_key, load_questions  # noqa: E402,E501
from hagstofa import llm  # noqa: E402
from hagstofa.pipeline import TOTAL_WORDS, answer, answer_from_table, pick_table  # noqa: E402
from hagstofa.retrieve import DEFAULT, build  # noqa: E402

RESULTS = ROOT / "eval" / "results"


def number_ok(text, expected):
    """expected: one number or a list (calc / two-table items): every one must be in the text."""
    return answer_ok(text, expected if isinstance(expected, list) else [expected])


def oracle_table(item):
    """The one table that answers the item, or None when it needs two (no single "right table" to hand over)."""
    tables = {a["table"] for a in item.get("calc", {}).get("args", [])} or {accepted(item)[0]}
    return tables.pop() if len(tables) == 1 and "all" not in item else None


def values_ok(table, selections, truth):
    """Every truth code was selected; a variable left out (eliminated) counts if the truth is its total."""
    for v in table["variables_is"]:
        want = truth.get(v["code"])
        if not want:
            continue
        got = selections.get(v["code"])
        if got is None:
            text = v["valueTexts"][v["values"].index(want[0])].strip().lower()
            if not (v.get("elimination") and text in TOTAL_WORDS):
                return False
        elif not set(want) <= set(got):
            return False
    return True


def cached_expected(questions, by_id):
    """{question: [expected numbers]}, kept in eval/results/expected.json between runs, so a rerun costs no
    PX-Web calls for these. Keyed by question + answer key, so editing a question's key refetches it.
    Delete the file to refetch everything (published figures for past years are rarely revised)."""
    path = RESULTS / "expected.json"
    cache = json.loads(path.read_text()) if path.exists() else {}

    def key(item):  # truth items keep the key format of earlier runs, so their cached numbers are reused
        spec = item["truth"] if "truth" in item else {k: item[k] for k in ("calc", "all") if k in item}
        return item["q"] + " " + json.dumps(spec, sort_keys=True)

    def fetch(table_path, truth):  # a timeout becomes None (not cached: retried next run)
        return timed(expected_value, table_path, truth)[0]

    missing = [item for item in questions if has_answer_key(item) and key(item) not in cache]
    if missing:
        print(f"fetching expected numbers for {len(missing)} questions from PX-Web ...")
        for item in missing:
            values = expected_numbers(item, lambda tid: by_id[tid]["path"], fetch)
            if None not in values:
                cache[key(item)] = values
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    out = {}
    for item in questions:
        value = cache.get(key(item)) if has_answer_key(item) else None
        out[item["q"]] = value if isinstance(value, list) or value is None else [value]  # old caches: one number
    return out


def timed(fn, *args):
    start = time.time()
    try:
        return fn(*args), None, time.time() - start
    except Exception as e:  # noqa: BLE001 - one bad answer (bad JSON, API error) shouldn't stop the comparison
        return None, f"{type(e).__name__}: {e}", time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="model names as the server knows them (ollama list)")
    ap.add_argument("-q", "--questions", default=str(ROOT / "eval" / "questions_e2e.jsonl"))
    ap.add_argument("--modes", default="pick,oracle,full")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("-n", type=int, help="only the first n questions")
    ap.add_argument("--reasoning", help='reasoning_effort for every call, e.g. "none" (default: LLM_REASONING)')
    args = ap.parse_args()
    modes = args.modes.split(",")
    questions = load_questions(args.questions)[: args.n]
    retriever = build(DEFAULT)
    by_id = {t["id"]: t for t in retriever.tables}
    candidates = [retriever.search(item["q"], k=args.k) for item in questions]
    RESULTS.mkdir(exist_ok=True)
    expected = {}
    if {"oracle", "full"} & set(modes):
        expected = cached_expected(questions, by_id)
    stamp = time.strftime("%Y%m%d-%H%M")
    if args.reasoning is not None:
        llm.REASONING = args.reasoning or None
    tag = f"_reasoning-{llm.REASONING}" if llm.REASONING else ""

    summary = []
    for model in args.models:
        llm.MODEL = model
        _, err, load = timed(llm.chat, "Svaraðu í einu orði.", "Halló")  # loads the model; not counted
        if err:
            print(f"{model}: {err}")
            continue
        print(f"\n== {model} (loaded in {load:.0f} s)")
        score = {m: {"ok": 0, "values": 0, "n": 0, "sec": 0.0, "errors": 0} for m in modes}
        out = open(RESULTS / f"{model.replace(':', '_').replace('/', '_')}{tag}_{stamp}.jsonl", "w", encoding="utf-8")
        for item, cands in zip(questions, candidates):
            accept, truth, exp = accepted(item), item.get("truth"), expected.get(item["q"])
            rec = {"q": item["q"], "accept": accept, "expected": exp}
            for mode in modes:
                s = score[mode]
                if mode == "pick":
                    t, err, sec = timed(pick_table, item["q"], cands)
                    ok = bool(t) and t["id"] in accept
                    rec["pick"] = {"picked": t and t["id"], "ok": ok,
                                   "in_candidates": any(c["id"] in accept for c, _ in cands)}
                else:
                    if not has_answer_key(item) or (mode == "oracle" and not oracle_table(item)):
                        continue  # no answer key; or oracle for a two-table question (no single right table)
                    if mode == "oracle":
                        r, err, sec = timed(answer_from_table, item["q"], by_id[oracle_table(item)])
                    else:
                        r, err, sec = timed(answer, item["q"], retriever, args.k)
                    r = r or {}
                    ok = number_ok(r.get("answer", ""), exp)
                    table = r.get("table")
                    v_ok = bool(truth) and bool(table) and table["id"] == accept[0] and values_ok(
                        table, r.get("selections", {}), truth)  # values are scored for one-cell items only
                    s["values"] += v_ok
                    calcs = [p.get("calculation") for p in r.get("parts", [])]
                    rec[mode] = {"ok": ok, "values_ok": v_ok, "tables": [p["table"]["id"] for p in r.get("parts", [])],
                                 "selections": [p["selections"] for p in r.get("parts", [])],
                                 "calculations": [c and c["text"] for c in calcs], "answer": r.get("answer")}
                s["n"] += 1
                s["ok"] += ok
                s["sec"] += sec
                s["errors"] += err is not None
                if err:
                    rec.setdefault("errors", {})[mode] = err
                print(f'  {mode:6s} {"OK " if ok else "-- "} {sec:5.1f}s  {item["q"][:70]}' + (f"  [{err[:60]}]" if err else ""))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.close()
        summary.append((model, score))

    print(f"\n{'model':22s}" + "".join(f"{m:>28s}" for m in modes))
    for model, score in summary:
        cells = []
        for m in modes:
            s = score[m]
            n = max(s["n"], 1)
            extra = f' val {s["values"] / n:4.0%}' if m != "pick" else ""
            cells.append(f'{s["ok"] / n:5.0%}{extra}  {s["sec"] / n:5.1f}s' + (f' err{s["errors"]}' if s["errors"] else ""))
        print(f"{model:22s}" + "".join(f"{c:>28s}" for c in cells))
    print("\npick: right table among the top-k | oracle/full: right number in the answer, val: right values chosen"
          f" | s: seconds per question | answers in {RESULTS}/")


if __name__ == "__main__":
    main()
