"""Which candidates should the table picker see? (31 Oct evening; search only, no LLM except cached translations)

    EMBED_FT=<fine-tuned model folder> python eval/experiments/candidate_union.py [sets...]

Run 8 showed the plain and the fine-tuned search tie end to end but miss different tables (the fine-tuned one finds
the 1989 wage index, LAU04000). Compared here, all through the bilingual search (English also searched in
Icelandic, from the translation cache):
  plain     raw + lemma + bge-m3, fused and checked (the demo)
  ft        the same with the fine-tuned bge-m3
  four      raw + lemma + both embeddings in one fusion
  union     plain's top k/2+ and ft's, interleaved, duplicates dropped: what a picker would see from both
Recall@k: the share of questions (parts, for two-table questions) with an accepted table in the first k.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from hagstofa import retrieve  # noqa: E402
from hagstofa.checker import CheckedRetriever  # noqa: E402

KS = (1, 3, 8, 12, 16)


def load(name):
    """[(question, [accepted ids])]: our sets (one table) and the organizers' format (parts with accepted)."""
    paths = {"dev": "eval/questions_dev.jsonl", "hand": "eval/questions.jsonl", "muni": "eval/questions_muni.jsonl",
             "real": "eval/questions_real.jsonl", "local": "eval/questions_localities.jsonl",
             "e2e": "eval/questions_e2e.jsonl", "dev_en": "eval/questions_dev_en.jsonl",
             "the60": os.path.join(os.environ.get("GAGNVIST_STARTER", ""), "questions/examples.jsonl"),
             "new41": "eval/questions_new.jsonl"}
    path = Path(paths[name]) if os.path.isabs(paths[name]) else ROOT / paths[name]
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        if "parts" in item:  # organizers' format: one entry per part, searched with the whole question
            for p in item["parts"]:
                out.append((item["question"], p.get("accepted") or [p["table"]]))
        else:
            t = item["table"]
            out.append((item["q"], t if isinstance(t, list) else [t]))
    return out


class Union:
    def __init__(self, a, b):
        self.a, self.b, self.tables = a, b, a.tables

    def search(self, question, k=8):
        ra, rb = self.a.search(question, k=k), self.b.search(question, k=k)
        seen, out = set(), []
        for pair in zip(ra, rb):
            for t, s in pair:
                key = (t["id"], t["path"])
                if key not in seen:
                    seen.add(key)
                    out.append((t, s))
        return out[:k]


def main():
    sets = sys.argv[1:] or ["dev", "hand", "muni", "real", "local", "e2e", "dev_en", "the60"]
    tables = retrieve.load_catalog()
    raw, lemma = retrieve.RawRetriever(tables), retrieve.LemmaRetriever(tables)
    plain_e = retrieve.EmbeddingRetriever(tables)
    ft_e = retrieve.EmbeddingRetriever(tables, model_name=os.environ["EMBED_FT"])

    def bilingual(members):
        return retrieve.BilingualRetriever(CheckedRetriever(retrieve.FusionRetriever(members)))

    configs = {"plain": bilingual([raw, lemma, plain_e]), "ft": bilingual([raw, lemma, ft_e]),
               "four": bilingual([raw, lemma, plain_e, ft_e])}
    configs["union"] = Union(configs["plain"], configs["ft"])
    print(f'{"set":8} {"n":>4}  ' + "  ".join(f'{c:>26}' for c in configs))
    print(f'{"":8} {"":>4}  ' + "  ".join(f'{"@" + "/@".join(map(str, KS)):>26}' for _ in configs))
    for name in sets:
        try:
            qs = load(name)
        except FileNotFoundError:
            print(f"{name}: no file")
            continue
        cells = []
        for c, r in configs.items():
            hits = {k: 0 for k in KS}
            for q, accept in qs:
                ids = [t["id"] for t, _ in r.search(q, k=max(KS))]
                rank = next((i + 1 for i, tid in enumerate(ids) if tid in accept), None)
                for k in KS:
                    hits[k] += bool(rank and rank <= k)
            cells.append("/".join(f"{round(100 * hits[k] / len(qs))}" for k in KS))
        print(f"{name:8} {len(qs):>4}  " + "  ".join(f"{cell:>26}" for cell in cells), flush=True)
    print("UNION_DONE")


if __name__ == "__main__":
    main()
