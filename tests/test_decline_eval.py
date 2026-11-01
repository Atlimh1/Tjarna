"""eval/decline_eval.py: a question we can't answer counts as declined only when no table was cited."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from decline_eval import NODATA, outcome  # noqa: E402

from hagstofa import guard  # noqa: E402


def test_guard_stop_is_a_decline():
    assert outcome(guard.Verdict("personal", "x"), {"answer": "..."}) == ("declined", "guard: personal")


def test_no_table_is_a_decline_and_a_cited_table_is_an_answer():
    ok = guard.Verdict("ok", "x")
    assert outcome(ok, {"answer": "Engin gögn", "parts": []}) == ("declined", "no table")
    parts = [{"table": {"id": "MAN00101"}}, {"table": {"id": "VIS01000"}}]
    assert outcome(ok, {"answer": "5", "parts": parts}) == ("answered", "MAN00101, VIS01000")


def test_the_nodata_set_is_readable():
    import json
    rows = [json.loads(line) for line in open(NODATA, encoding="utf-8")]
    assert len(rows) == 16 and all(r["question"] and r["id"] for r in rows)


def test_real2_verdicts():
    import real2_eval as r2
    assert r2.verdict("decline", "declined", "no table", []) == "ok"
    assert r2.verdict("decline", "answered", "MAN00101", []) == "answered (should decline)"
    assert r2.verdict("answer", "answered", "SAM03105", ["SAM03105"]) == "ok"
    assert r2.verdict("answer", "answered", "X1, SAM03105", ["SAM03105"]) == "ok"
    assert r2.verdict("answer", "answered", "X1", ["SAM03105"]) == "other table"
    assert r2.verdict("answer", "declined", "no table", ["SAM03105"]) == "declined (should answer)"
    assert r2.verdict("partial", "declined", "guard: offtopic", ["X"]) == "ok"
