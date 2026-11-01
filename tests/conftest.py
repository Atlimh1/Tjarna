"""Shared test settings: no test may call a real language model."""
import pytest

from hagstofa import suggest


@pytest.fixture(autouse=True)
def no_model_written_suggestions(monkeypatch):
    """Suggestions after "no data" are written by the model in production; tests use the fallback (the tables' own
    questions or titles) unless they replace llm.chat themselves and switch it back on."""
    monkeypatch.setattr(suggest, "WRITE", False)
