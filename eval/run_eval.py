"""Score the system against eval/questions.jsonl.

Each line: {"q": question, "table": [acceptable table ids], "truth": {var_code: [value_code]}}
"table" lists every table that answers the question (the first is the one `truth` refers to).
"truth" is optional; --e2e scores only questions with an answer key: "truth" (one cell), "calc" (an operation
over cells: {"op", "args": [{"table", "truth"}, ...]}) or "all" (several cells, every one must be in the answer).
Expected numbers are fetched live from the API, so answers stay correct as tables update.

    .venv/bin/python eval/run_eval.py                    # retrieval only: is the right table in top-k? (no LLM)
    .venv/bin/python eval/run_eval.py -r embed           # pick a retriever: raw, lemma, embed, fusion
    .venv/bin/python eval/run_eval.py -r all             # compare all retrievers
    .venv/bin/python eval/run_eval.py --e2e              # full pipeline: right table picked, right number in answer
    .venv/bin/python eval/run_eval.py --pick -q dev      # LLM table choice among the top-k only (no truth, no API calls)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hagstofa import px  # noqa: E402
from hagstofa.pipeline import answer, compute, pick_table  # noqa: E402
from hagstofa.retrieve import DEFAULT, RETRIEVERS, build  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
QUESTION_SETS = {
    "hand": QUESTIONS,                               # 36 hand-written; SYNONYMS were tuned on these
    "dev": ROOT / "eval" / "questions_dev.jsonl",    # generated held-out, ~30% of tables: tune freely
    "test": ROOT / "eval" / "questions_test.jsonl",  # generated held-out, ~70%: frozen, never tune on it
}


def load_questions(path=QUESTIONS):
    return [json.loads(l) for l in open(QUESTION_SETS.get(path, path), encoding="utf-8") if l.strip()]


def accepted(item):
    return item["table"] if isinstance(item["table"], list) else [item["table"]]


def evaluate_retrieval(retriever, questions, k=8):
    """One row per question: {q, accept, ids (top-k), rank (1-based) or None, folder_rank}.
    folder_rank: first result from the same subject folder as an accepted table. Generated questions
    list only their source table, so this shows how often search lands in the right area anyway."""
    folder = {t["id"]: t["path"].rsplit("/", 1)[0] for t in retriever.tables}
    rows = []
    for item in questions:
        accept = accepted(item)
        ids = [t["id"] for t, _ in retriever.search(item["q"], k=k)]
        rank = next((i + 1 for i, tid in enumerate(ids) if tid in accept), None)
        want = {folder.get(a) for a in accept}
        folder_rank = next((i + 1 for i, tid in enumerate(ids) if folder.get(tid) in want), None)
        rows.append({"q": item["q"], "accept": accept, "ids": ids, "rank": rank, "folder_rank": folder_rank})
    return rows


def summary(rows, k):
    n = len(rows)
    top = lambda key, j: sum(r[key] is not None and r[key] <= j for r in rows)  # noqa: E731
    return (f'top1 {top("rank", 1)}/{n} ({top("rank", 1) / n:.0%}), top3 {top("rank", 3) / n:.0%}, '
            f'top{k} {top("rank", k)}/{n} ({top("rank", k) / n:.0%}) | right folder top{k} {top("folder_rank", k) / n:.0%}')


NUMBER_RE = re.compile(
    r'(?<![\d.,])-?(?:\d{1,3}(?:[.  ]\d{3})+(?:,\d+)?'  # Icelandic grouping: 40.402 / 394 324 / 1.234,5
    r'|\d{1,3}(?:,\d{3})+(?:\.\d+)?'                           # English grouping: 1,234,567 / 1,234.5
    r'|\d+(?:[.,]\d+)?)'                                       # plain or decimal: 2025 / 3,8 / 4.0
)


def numbers_in(text):
    """Numbers in Icelandic or English formatting. Ambiguous '40.402' / '1,834' yield both readings."""
    out = []
    text = re.sub("[  ]", " ", text)  # thin / narrow no-break space: gpt-oss writes "394 324"
    for tok in NUMBER_RE.findall(text):
        sign = -1 if tok.startswith("-") else 1
        tok = tok.lstrip("-").replace(" ", " ")
        if re.fullmatch(r"\d{1,3}[.,]\d{3}", tok):
            out += [sign * float(re.sub(r"[.,]", "", tok)), sign * float(tok.replace(",", "."))]
        elif re.fullmatch(r"\d+,\d+", tok):  # decimal comma
            out.append(sign * float(tok.replace(",", ".")))
        elif re.fullmatch(r"\d{1,3}(?:[. ]\d{3})+(?:,\d+)?", tok):  # Icelandic grouping
            out.append(sign * float(re.sub(r"[. ]", "", tok).replace(",", ".")))
        else:  # English grouping or decimal point
            out.append(sign * float(tok.replace(",", "")))
    return out


def expected_value(table_path, truth):
    values = px.query(table_path, truth)["value"]
    return values[0] if len(values) == 1 else None


def has_answer_key(item):
    return any(key in item for key in ("truth", "calc", "all"))


def expected_numbers(item, path_of, fetch=None):
    """The numbers a right answer must contain, fetched live. Three kinds of item (eval/build_e2e.py):
    truth = one cell; calc = an operation over cells, computed with the pipeline's own compute();
    all = several cells, possibly from different tables (two-table questions): every one must appear.
    None in the list = a cell couldn't be fetched (the item can't be scored)."""
    fetch = fetch or expected_value
    cell = lambda spec: fetch(path_of(spec["table"]), spec["truth"])  # noqa: E731
    if "truth" in item:
        return [fetch(path_of(accepted(item)[0]), item["truth"])]
    if "calc" in item:
        values = [cell(a) for a in item["calc"]["args"]]
        return [None if None in values else compute(item["calc"]["op"], values)]
    return [cell(a) for a in item["all"]]


def answer_ok(text, expected):
    """Every expected number appears in the text (Icelandic or English formatting, 0.5% or 0.05 tolerance)."""
    found = numbers_in(text or "")
    return bool(expected) and None not in expected and all(
        any(abs(n - e) <= max(abs(e) * 0.005, 0.05) for n in found) for e in expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--pick", action="store_true", help="score the LLM's table choice among the top-k")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("-r", "--retriever", default=DEFAULT, choices=RETRIEVERS + ["all", "bilingual"])
    ap.add_argument("-q", "--questions", default="hand", help="hand, dev, test, or a path to a .jsonl file")
    args = ap.parse_args()
    questions = load_questions(args.questions)
    if args.questions == "test":
        print("TEST SET: report these numbers, but don't change anything because of them.\n")

    if args.retriever == "all":
        for name in RETRIEVERS:
            print(f"{name:8s} {summary(evaluate_retrieval(build(name), questions, args.k), args.k)}")
        return

    retriever = build(args.retriever)
    by_id = {t["id"]: t for t in retriever.tables}
    picked = correct = n_e2e = 0
    pick_ok = pick_none = 0
    rows = evaluate_retrieval(retriever, questions, args.k)

    for item, row in zip(questions, rows):
        rank = row["rank"]
        line = f'{"top" + str(rank) if rank else "MISS":>5}  {item["q"]}'
        if not rank:
            line += f'\n       want {row["accept"]}, got {row["ids"][:3]}'
        if args.pick:
            t = pick_table(item["q"], retriever.search(item["q"], k=args.k))
            pick_ok += bool(t) and t["id"] in row["accept"]
            pick_none += t is None
            line += f'\n       LLM picked {t and t["id"]} {"OK" if t and t["id"] in row["accept"] else "WRONG"}'
        if args.e2e and has_answer_key(item):
            n_e2e += 1
            r = answer(item["q"], retriever, k=args.k)
            got_table = r.get("table", {}).get("id")
            picked += got_table in row["accept"]
            exp = expected_numbers(item, lambda tid: by_id[tid]["path"])
            ok = answer_ok(r["answer"], exp)
            correct += ok
            line += f'\n       picked={got_table} expected={exp} {"OK" if ok else "WRONG"}\n       {r["answer"][:200]!r}'
        print(line)

    print(f"\nretrieval ({args.retriever}): {summary(rows, args.k)}")
    if args.pick:
        n, hits = len(rows), sum(r["rank"] is not None for r in rows)
        print(f"LLM pick: right table {pick_ok}/{n} ({pick_ok / n:.0%}); {pick_ok}/{hits} when it was in the top{args.k}; "
              f"no table chosen {pick_none}")
    if args.e2e:
        print(f"end-to-end ({n_e2e} questions with an answer key): table picked {picked}/{n_e2e}, correct number {correct}/{n_e2e}")


if __name__ == "__main__":
    main()
