"""Thin wrappers around Miðeind's Icelandic tools, as used by the pipeline.

  correct(text)            GreynirCorrect: fix spelling/accents/capitals ("Kopavogi" -> "Kópavogi")
  tokens_with_lemmas(text) Tokenizer + BÍN: what retrieval sees
  entities(text)           Greynir parse: places, time expressions and noun phrases
  parse_tree(text)         Greynir's sentence tree, as text
  inflect(text, case)      Greynir: each noun phrase in the requested case (nf/þf/þgf/ef)
"""
import functools

import tokenizer

from .retrieve import _lemmas_of

TIME_CATEGORIES = {"ártal", "dagsafs", "dagsfast", "tímapunktur", "tímapunkturafs", "tímapunkturfast", "tími"}
TIME_NODES = ("ADVP-DATE-ABS", "ADVP-DATE-REL", "ADVP-TIMESTAMP-ABS", "ADVP-TIMESTAMP-REL", "ADVP-DUR-ABS", "ADVP-DUR-REL")
CASE_ATTR = {"nf": "nominative_np", "þf": "accusative_np", "þgf": "dative_np", "ef": "genitive_np"}


@functools.cache
def _greynir():
    from reynir import Greynir  # loads grammar on first use

    return Greynir()


def correct(text):
    """(corrected text, [human-readable notes]) using GreynirCorrect."""
    import reynir_correct

    result = reynir_correct.check_single(text)
    if result is None:
        return text, []
    return result.tidy_text, [a.text for a in result.annotations]


def tokens_with_lemmas(text):
    """[(token text, token kind, [BÍN lemmas])] for every token, the way retrieval sees them."""
    out = []
    for tok in tokenizer.tokenize(text):
        if not tok.txt:
            continue
        lem = list(_lemmas_of(tok.txt)) if tok.kind == tokenizer.TOK.WORD else []
        out.append((tok.txt, tokenizer.TOK.descr[tok.kind], lem))
    return out


def _parse(text):
    s = _greynir().parse_single(text)
    return s if s is not None and s.tree is not None else None


def entities(text):
    """Places (proper nouns), time expressions and noun phrases found by Greynir."""
    s = _parse(text)
    if s is None:
        return {"places": [], "times": [], "noun_phrases": []}
    places = [t.lemma for t in s.terminals if t.category == "no" and t.lemma[:1].isupper()]
    times = [t.text for t in s.terminals if t.category in TIME_CATEGORIES]
    for node in TIME_NODES:
        times += [m.text for m in s.tree.all_matches(node) if m.text not in times]
    return {"places": places, "times": times, "noun_phrases": [m.text for m in s.tree.all_matches("NP")]}


def parse_tree(text):
    s = _parse(text)
    return s.tree.view if s is not None else "(Greynir náði ekki að greina setninguna)"


def inflect(text, case):
    """Markdown list: each noun phrase -> the same phrase in `case`."""
    s = _parse(text)
    if s is None:
        return "_Greynir náði ekki að greina setninguna._"
    lines = [f"- {m.text} → **{getattr(m, CASE_ATTR[case])}**" for m in s.tree.all_matches("NP")]
    return "\n".join(lines) or "_Engir nafnliðir._"
