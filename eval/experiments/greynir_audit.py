"""Experiment 3: what GreynirCorrect changes in questions that were already written correctly (dev).

    .venv/bin/python eval/experiments/greynir_audit.py

dev's 270 questions are clean Icelandic, so every change is a false correction. For each changed word: the
category (capitals only, accents only, a capitalised name, another form of the same word = inflection, a
different word, or words dropped/merged/split), for GreynirCorrect's raw output and for what is left after
safe_correction() (in correct_question since 31 Oct: a change is kept only where the typed word is not a word).

RESULTS (31 Oct; 246 Icelandic dev questions, the other 24 are English and skipped):
  GreynirCorrect raw: 25 changed. Words: 18 number ranges re-typeset ("20-24" -> "20–24", "2002-2003?" ->
  "2002–2003?"), 6 split or merged ("1993-2010" -> "1993 — 2010", "1985/1986" -> "1985 1986", "ennþá" -> "enn þá"),
  4 capitals ("norðvesturkjördæmi" -> "Norðvesturkjördæmi", "kjörsókn" -> "Kjörsókn"), 1 inflection ("sent" -> "send").
  No names changed, no rare words replaced.
  After safe_correction() (today's correct_question): 4 changed, all capitals only (the 4 above). Harmless for
  search (raw and lemma lowercase; the checker's names() may even see "Norðvesturkjördæmi" as a place).
  Decision: no change. The 31 Oct safe_correction already removed every harmful false correction on dev;
  protecting names isn't needed (GreynirCorrect changed none).
"""
import difflib
import re
import sys
from collections import Counter

from recall import ROOT, load

sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import _lemmas_of, correct_question, fold, is_english, safe_correction  # noqa: E402


def raw_greynir(question):
    """GreynirCorrect's own output (cached in data/corrections.sqlite by correct_question)."""
    import sqlite3

    from hagstofa.retrieve import CORRECTIONS
    correct_question(question)  # fills the cache
    with sqlite3.connect(CORRECTIONS) as db:
        row = db.execute("SELECT fixed FROM fix WHERE q=?", (question,)).fetchone()
    return row[0] if row else question


def category(old, new, first):
    core_old, core_new = re.sub(r"^\W+|\W+$", "", old), re.sub(r"^\W+|\W+$", "", new)
    if core_old == core_new:
        return "punctuation"
    if core_old.lower() == core_new.lower():
        return "capitals"
    if fold(core_old.lower()) == fold(core_new.lower()):
        return "accents"
    if core_old[:1].isupper() and not first:
        return "name"
    if set(_lemmas_of(core_old.lower())) & set(_lemmas_of(core_new.lower())):
        return "inflection"
    return "other word"


def changes(question, fixed):
    a, b = question.split(), fixed.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            out += [(category(o, n, i1 + k == 0), o, n) for k, (o, n) in enumerate(zip(a[i1:i2], b[j1:j2]))]
        else:
            out.append(("dropped/merged/split", " ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def main():
    questions = [q["q"] for q in load("dev") if not is_english(q["q"])]
    for label, fix in (("GreynirCorrect raw", raw_greynir),
                       ("after safe_correction (today's correct_question)", lambda q: safe_correction(q, raw_greynir(q)))):
        per_q = [(q, changes(q, fix(q))) for q in questions]
        changed = [(q, c) for q, c in per_q if c]
        cats = Counter(cat for _, c in changed for cat, _, _ in c)
        print(f"\n== {label}: {len(changed)} of {len(questions)} correct questions changed; words by category:")
        for cat, n in cats.most_common():
            examples = [f"{o} -> {n_}" for _, c in changed for k, o, n_ in c if k == cat][:8]
            print(f"   {cat:<22} {n:>3}   e.g. {'; '.join(examples)}")
        for q, c in changed[:40]:
            print(f"     {q}\n        " + "; ".join(f"[{k}] {o} -> {n}" for k, o, n in c))


if __name__ == "__main__":
    main()
