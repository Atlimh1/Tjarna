"""Experiment 1: query rewriting inside the split call (pipeline.decompose_rewrite, switch QUERY_REWRITE=1).

    LLM_BASE_URL=... LLM_MODEL=... .venv/bin/python eval/experiments/query_rewrite.py [--sets dev,real] [--no-timing]

The split call (DECOMPOSE_SYSTEM) also returns a clean version of each sub-question: standard spelling with
accents, full words for abbreviations, relative dates made explicit from today's date; English stays English.
Same call, a longer JSON. The search then runs on the question as typed AND the clean version, fused by RRF
(pipeline.search_with_rewrite); pick, values and the answer keep the user's wording.

For each question in dev, the no-accent and slip halves of dev_typo, and real: recall @1/@3/@8/@12 of `checked` for
  original   the question as typed (today's search)
  rewritten  only the clean version(s) (several parts: fused)
  fused      original + clean, by RRF (what QUERY_REWRITE=1 does)
and the extra seconds per question: the rewrite call minus a plain decompose() call on the same question (both
timed; --no-timing skips the plain call), plus the second search. Rewrites are cached in
eval/results/rewrites-<model>.jsonl, so a rerun costs only the searches.
Needs an LLM. Not run yet (31 Oct): no model reachable from the cloud session; run it on JURECA.

RESULTS: (fill in from a run)
"""
import argparse
import json
import os
import sys
import time

from recall import ROOT, question_sets, rank, recall

sys.path.insert(0, str(ROOT))
from hagstofa import llm, pipeline  # noqa: E402
from hagstofa.retrieve import build, rrf  # noqa: E402


def cache_path():
    model = os.environ.get("LLM_MODEL", llm.MODEL).replace("/", "_").replace(":", "_")
    return ROOT / "eval/results" / f"rewrites-{model}.jsonl"


def load_cache(path):
    if not path.exists():
        return {}
    return {r["q"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def rewrite(question, cache, out, timing):
    if question in cache:
        return cache[question]
    t = time.time()
    pairs = pipeline.decompose_rewrite(question)
    rewrite_s = time.time() - t
    plain_s = None
    if timing:
        t = time.time()
        pipeline.decompose(question)
        plain_s = time.time() - t
    row = {"q": question, "pairs": pairs, "rewrite_s": rewrite_s, "plain_s": plain_s}
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    cache[question] = row
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", help="comma-separated: dev, no accents, slips, real (default: all)")
    ap.add_argument("--no-timing", action="store_true", help="skip the plain decompose() call used for timing")
    args = ap.parse_args()
    sets = question_sets()
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets.split(",")}
    checked = build("checked")
    path = cache_path()
    path.parent.mkdir(exist_ok=True)
    cache = load_cache(path)
    print(f"rewrites cached in {path} ({len(cache)} so far); model {os.environ.get('LLM_MODEL', llm.MODEL)}")
    with open(path, "a", encoding="utf-8") as out:
        for name, items in sets.items():
            ranks = {"original": [], "rewritten": [], "fused": []}
            extra, search_s, examples = [], [], []
            for item in items:
                row = rewrite(item["q"], cache, out, not args.no_timing)
                cleans = [c for _, c in row["pairs"] if c]
                original = checked.search(item["q"], k=20)
                t = time.time()
                rewritten = rrf([checked.search(c, k=20) for c in cleans], k=20) if cleans else original
                search_s.append(time.time() - t)
                fused = rrf([original, *[checked.search(c, k=20) for c in cleans]], k=20) if cleans else original
                for key, found in (("original", original), ("rewritten", rewritten), ("fused", fused)):
                    ranks[key].append(rank([t_["id"] for t_, _ in found[:12]], item))
                if row["plain_s"] is not None:
                    extra.append(row["rewrite_s"] - row["plain_s"])
                if ranks["fused"][-1] != ranks["original"][-1] and len(examples) < 12:
                    examples.append(f'    {ranks["original"][-1]} -> {ranks["fused"][-1]}  {item["q"]}  =>  {cleans}')
            print(f"\n{name} (n={len(items)}): @1/@3/@8/@12")
            for key, r in ranks.items():
                print(f"  {key:<10} {recall(r)}")
            if extra:
                print(f"  extra seconds per question: rewrite vs plain split call {sum(extra) / len(extra):+.2f} s, "
                      f"second search {sum(search_s) / len(search_s):.2f} s")
            print("  changed by fusion (original rank -> fused rank):\n" + "\n".join(examples), flush=True)


if __name__ == "__main__":
    main()
