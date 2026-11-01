"""Experiment (2026-10-25): LLM table picking variants, on dev, with qwen3:4b on a small GPU (GTX 1650).

Baseline (run_eval.py --pick): pick among top-8 with a "-1 = none fits" option -> 28%; top-3 -> 29%,
but the model refused ("none") on 94 of 270 questions, 57 of them with the right table in the list.
This script: forced choice (no "none" option) among the top-k, broken down by whether the three
retrievers (raw, lemma, embed) agree on #1.

Results (k=3, forced choice):
    group          n    fusion #1   forced LLM top3
    all 3 agree    38   87%         61%     <- LLM breaks a near-certain #1
    2 of 3 agree  108   45%         44%
    all differ    124   25%         32%     <- the only group where the LLM helps
    TOTAL         270   42%         41%     (vs 29% with the "-1 = none" option: refusals were the main loss)
Cascade "take #1 if any two retrievers agree, else forced LLM" ~= (33+49+40)/270 = 45%: +3, inside noise.
The 5 all-agree misses are real errors, not label artifacts; 3 of them a value check would catch:
France / Americans / the year 2000 + Reykjavik are not values of the table that was picked.

    LLM_BASE_URL=http://<gpu>:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M \
        .venv/bin/python eval/experiments/pick_strategies.py -k 3
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from hagstofa import llm  # noqa: E402
from hagstofa.retrieve import build  # noqa: E402
from run_eval import accepted, load_questions  # noqa: E402

FORCED_SYSTEM = (
    "Þú velur töflu úr gagnagrunni Hagstofu Íslands sem svarar spurningu notanda best. "
    'Veldu alltaf eina töflu. Svaraðu eingöngu með JSON: {"table": <númer>}.'
)


def forced_pick(question, candidates):
    listing = "\n".join(
        f'[{i}] {t["id"]}: {t["title_is"]} | breytur: {", ".join(v["text"] for v in t["variables_is"])}'
        for i, (t, _) in enumerate(candidates)
    )
    idx = llm.chat_json(FORCED_SYSTEM, f"Spurning: {question}\n\nTöflur:\n{listing}").get("table", 0)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    return candidates[idx][0] if 0 <= idx < len(candidates) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("-q", "--questions", default="dev")
    args = ap.parse_args()
    R = {n: build(n) for n in ("raw", "lemma", "embed", "fusion")}
    titles = {t["id"]: t["title_is"] for t in R["fusion"].tables}
    g = defaultdict(lambda: {"n": 0, "top1": 0, "llm": 0, "none": 0})
    for item in load_questions(args.questions):
        acc = accepted(item)
        tops = [R[n].search(item["q"], k=1)[0][0]["id"] for n in ("raw", "lemma", "embed")]
        group = {3: "all 3 agree", 2: "2 of 3 agree", 1: "all differ"}[max(map(tops.count, tops))]
        cands = R["fusion"].search(item["q"], k=args.k)
        t = forced_pick(item["q"], cands)
        s = g[group]
        s["n"] += 1
        s["top1"] += cands[0][0]["id"] in acc
        s["llm"] += bool(t) and t["id"] in acc
        s["none"] += t is None
        if group == "all 3 agree" and cands[0][0]["id"] not in acc:
            print(f'all-agree miss: {item["q"]}\n   want {acc[0]}: {titles[acc[0]]}\n'
                  f'   got  {cands[0][0]["id"]}: {cands[0][0]["title_is"]}')
    tot = {key: sum(s[key] for s in g.values()) for key in ("n", "top1", "llm", "none")}
    for name, s in [*sorted(g.items()), ("TOTAL", tot)]:
        print(f'{name:13s} n={s["n"]:3d}  fusion #1 {s["top1"] / s["n"]:.0%}  '
              f'forced LLM top{args.k} {s["llm"] / s["n"]:.0%}  unparsable {s["none"]}')


if __name__ == "__main__":
    main()
