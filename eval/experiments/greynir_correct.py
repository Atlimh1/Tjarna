"""Experiment (2026-10-24): spelling correction (GreynirCorrect, English skipped) before retrieval, and accent
folding in raw ("haskolanams" = "háskólanáms"). Measured on dev, misspelled copies of dev
(eval/make_typos.py: no accents / 1-2 slips) and the user's real questions. fusion top1/top3/top8:

                       before      corrected   folded      both (KEPT, now the default)
  dev clean (is, 246)  44/67/83    44/67/82    44/67/82    45/67/82
  dev clean (en, 24)   38/62/75    38/62/75    38/62/75    38/62/75   (correcting English cost 4 pts top8 -> skipped)
  dev noaccent (is)    33/58/73    36/58/77    37/61/78    40/63/77
  dev typo (is)        41/64/82    43/66/82    41/65/80    44/66/82
  real (11)            36/36/64    27/27/64    27/36/64    18/27/55
  test (627, once)     46/67/82 -> 45/67/82 with both (no typos there: no change expected)

Findings: ordinary typos cost little (char n-grams and bge-m3 tolerate them); missing accents cost 11 points
because BÍN doesn't know "bua"/"islandi". Correction and folding fix different words (correction: short ones,
capital letters; folding: long multi-accent words), so together they recover 7 of the 11 points.
The real set drop, per question: þjóðkyrkjunni miss->7 (the fix working); "algengasta" now matches
"Algengustu krabbamein" (the typo had hidden that); "Reykjavík" capitalised fires the place boost on a
nationwide table; Stokkseyri was a borderline rank 4 (members rank it 11-48; it's a locality past the
40-value cap). None of the three losses is the correction's fault; all are known retrieval weaknesses.

    .venv/bin/python eval/experiments/greynir_correct.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from hagstofa import retrieve as R  # noqa: E402
from run_eval import evaluate_retrieval, load_questions, summary  # noqa: E402


class Unfolded(R.RawRetriever):
    """raw as it was before accent folding."""

    def __init__(self, tables):
        R._Base.__init__(self, tables)
        self.vec = R.TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(R.raw_table_text(t) for t in self.tables)

    def scores(self, question):
        return R.linear_kernel(self.vec.transform([question.lower()]), self.matrix)[0]


def main():
    typo = load_questions(str(ROOT / "eval" / "questions_dev_typo.jsonl"))
    dev = load_questions("dev")
    sets = {
        "dev clean (is)": [q for q in dev if q.get("lang") != "en"],
        "dev clean (en)": [q for q in dev if q.get("lang") == "en"],
        "dev noaccent (is)": [q for q in typo if q["typo_style"] == "noaccent" and q.get("lang") != "en"],
        "dev typo (is)": [q for q in typo if q["typo_style"] == "typo" and q.get("lang") != "en"],
        "dev typo (en)": [q for q in typo if q["typo_style"] == "typo" and q.get("lang") == "en"],
        "real": load_questions(str(ROOT / "eval" / "questions_real.jsonl")),
    }
    tables = R.load_catalog()
    lemma, embed = R.build("lemma", tables), R.build("embed", tables)
    variants = {
        "before": (R.FusionRetriever([Unfolded(tables), lemma, embed]), False),
        "corrected": (R.FusionRetriever([Unfolded(tables), lemma, embed]), True),
        "folded": (R.FusionRetriever([R.RawRetriever(tables), lemma, embed]), False),
        "both (default)": (R.FusionRetriever([R.RawRetriever(tables), lemma, embed]), True),
    }
    t = time.time()
    for qs in sets.values():
        for item in qs:
            R.correct_question(item["q"])  # fills data/corrections.sqlite
    print(f"corrections ready ({time.time() - t:.0f} s)")
    for name, qs in sets.items():
        for vname, (r, correct) in variants.items():
            r.correct = correct
            print(f"{name:18s} n={len(qs):3d} {vname:15s} {summary(evaluate_retrieval(r, qs, 8), 8)}", flush=True)


if __name__ == "__main__":
    main()
