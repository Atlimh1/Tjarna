"""Experiment 2: put Icelandic letters back without a model, before GreynirCorrect (retrieve.restore_accents).

    .venv/bin/python eval/experiments/accents.py            # modes 0, 1, 2 on every set (CPU, ~15 min first time)

Rule: a word with no Icelandic letters gets the one COMMON word (Icegrams count >= 100) among the real BÍN entries
(bin_id > 0; islenska otherwise guesses "öhöflegur") and Icelandic catalog words that fold() to it.
  mode 1 = only words BÍN doesn't know ("hvad" -> "hvað", "bua" -> "búa", "i" -> "í")
  mode 2 = also known but rare words whose twin is >= 50x as common ("hatt" -> "hátt", "ara" -> "ára", and
           "arid" -> "árið" when two candidates exist and one dominates)
Switch: ACCENTS=1 or ACCENTS=2 (off by default). English questions are skipped (is_english), and so is any question
with an Icelandic letter in it (see the results: there an unaccented word is a typo, not a lost accent).

Measured: recall @1/@3/@8/@12 of the `checked` retriever, and for the no-accent questions the share of words that
match the correctly written original (`orig`) after restoration (before GreynirCorrect).

RESULTS (31 Oct, checked; dev = 270, the no-accent and slip halves of dev_typo = 270 each, real = 11):
  word accuracy on the no-accent questions: as typed 49.6%, mode 1 88.5%, mode 2 95.6%
    mode 1 misses: arid->árið x73 (two candidates, arið/árið: declines), ad/vid/med (að/við/með: "ad", "vid", "med"
    are real BÍN entries), ara->ára, hatt->hátt; mode 2 fixes most of those but not med->með (BÍN "med" is common)
                       @1/@3/@8/@12    as typed      mode 1        mode 2        mode 1, unaccented questions only
    dev                                51/71/86/90   51/71/86/90   51/71/86/90   51/71/86/90
    no accents                         43/67/84/90   50/70/86/90   50/70/86/90   50/70/86/90
    slips                              50/71/86/89   50/71/85/88   50/71/85/88   50/71/86/89
    real                               45/45/64/73   45/45/64/73   45/45/64/73   45/45/64/73
  No accents +7 points @1 (40 questions better, 15 worse), beyond dev's +-3 noise; dev and real unchanged.
  Mode 2's better word accuracy doesn't show in search (GreynirCorrect and the accent-folding raw search already
  cover "ára", "árið"), and it moves 4 more questions down: mode 1 it is.
  The 2 slips lost were typos read as missing accents: "lanin" (launin) -> "lánin", "skatar" (skattar) -> "skátar".
  Both questions had other Icelandic letters, so restoration now runs only on questions with none (last column):
  slips back to as typed, the gain kept. Decision: switch ACCENTS=1 on (with that rule, now in restore_accents).
"""
import re
import sys
import time

from recall import ROOT, question_sets, rank, recall

sys.path.insert(0, str(ROOT))
from hagstofa import retrieve  # noqa: E402
from hagstofa.retrieve import build, restore_accents  # noqa: E402


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower())


def word_accuracy(items, mode):
    """Share of words equal to the original's (the no-accent set keeps it in `orig`), and examples of misses."""
    right = total = 0
    misses = []
    for item in items:
        got, want = words(restore_accents(item["q"], mode)), words(item["orig"])
        if len(got) != len(want):
            continue
        for g, w in zip(got, want):
            total += 1
            right += g == w
            if g != w and len(misses) < 400:
                misses.append((g, w))
    return right / max(total, 1), misses


def main():
    sets = question_sets()
    checked = build("checked")
    print("word accuracy on the no-accent questions (restored word == original word):")
    for mode in (0, 1, 2):
        acc, misses = word_accuracy(sets["no accents"], mode)
        common = {}
        for g, w in misses:
            common[(g, w)] = common.get((g, w), 0) + 1
        top = sorted(common.items(), key=lambda kv: -kv[1])[:12]
        print(f"  mode {mode}: {100 * acc:.1f}%   most common misses (got -> wanted): "
              + ", ".join(f"{g}->{w} x{n}" for (g, w), n in top), flush=True)
    print("\nrecall @1/@3/@8/@12 (checked):")
    ranks = {}
    for mode in (0, 1, 2):
        retrieve.ACCENTS = str(mode)
        for name, items in sets.items():
            t = time.time()
            ranks[mode, name] = [rank([x["id"] for x, _ in checked.search(i["q"], k=12)], i) for i in items]
            print(f"  mode {mode}  {name:<11} {recall(ranks[mode, name])}   ({time.time() - t:.0f} s)", flush=True)
    for mode in (1, 2):
        for name, items in sets.items():
            changed = [(i["q"], a, b) for i, a, b in zip(items, ranks[0, name], ranks[mode, name]) if a != b]
            better = sum((b or 99) < (a or 99) for _, a, b in changed)
            print(f"\nmode {mode} vs 0 on {name}: {better} better, {len(changed) - better} worse")
            for q, a, b in changed[:15]:
                print(f"    {a} -> {b}  {q}  |  {restore_accents(q, mode)}")


if __name__ == "__main__":
    main()
