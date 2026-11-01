import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import finetune_questions  # noqa: E402
from finetune_embed import batches  # noqa: E402


def test_batches_never_repeat_a_table_and_use_every_pair():
    pairs = [(f"q{t}-{i}", f"T{t}", f"text {t}") for t in range(10) for i in range(t % 4 + 1)]  # 1-4 questions per table
    out = list(batches(pairs, 4, random.Random(0)))
    assert all(len({p[1] for p in b}) == len(b) for b in out)  # one question per table in a batch: no false negatives
    assert all(len(b) >= 2 for b in out)  # a batch of 1 has nothing to compare against
    used = [p for b in out for p in b]
    assert len(used) == len(set(used)) and len(pairs) - len(used) <= 4  # each pair once; at most a few singletons left


def test_tables_of_skips_parts_without_a_table():
    # questions_mixed.jsonl: parts are sub-questions with no table (the job failed on them, 31 Oct)
    assert finetune_questions.tables_of({"q": "x", "parts": [{"q": "a"}, {"q": "b", "lang": "en"}]}) == []
    assert finetune_questions.tables_of({"parts": [{"table": "MAN1", "accepted": ["MAN1", "MAN2"]}, {"table": "VIS1"}]}) == ["MAN1", "MAN2", "VIS1"]
