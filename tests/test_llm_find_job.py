"""llm.find_model (LLM_FIND_JOB): a model whose JURECA job ended is replaced by another running job of that name."""
import subprocess
from types import SimpleNamespace as NS

import pytest

from hagstofa import llm


def test_running_nodes_read_the_queue_latest_end_first(monkeypatch):
    out = "2026-11-01T18:30:00 jrc0352\n2026-11-01T20:45:00 jrc0410\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: NS(stdout=out))
    assert llm.running_nodes("gemma-vllm") == ["jrc0410", "jrc0352"]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("squeue")))
    assert llm.running_nodes("gemma-vllm") == []  # not on a cluster: nothing to move to


def test_a_lost_model_moves_to_another_running_job_and_the_call_is_tried_again(monkeypatch):
    calls = []

    def fake_chat(system, user, json_mode, temperature):
        calls.append(str(llm._client.base_url))
        if "jrc0352" in calls[-1]:  # the old job: gone
            raise llm.IncompleteReply("the model stopped mid-reply (abort)")
        return "svar"
    monkeypatch.setattr(llm, "FIND_JOB", "gemma-vllm")
    monkeypatch.setattr(llm, "_client", llm.OpenAI(base_url="http://jrc0352i:8000/v1", api_key="k"))
    monkeypatch.setattr(llm, "_chat", fake_chat)
    monkeypatch.setattr(llm, "running_nodes", lambda job=None: ["jrc0410"])
    monkeypatch.setattr(llm.OpenAI, "models", property(lambda self: NS(list=lambda: [])), raising=False)
    assert llm.chat("s", "u") == "svar"
    assert "jrc0352" in calls[0] and "jrc0410" in calls[1] and "jrc0410" in str(llm._client.base_url)


def test_without_find_job_a_lost_model_is_an_error(monkeypatch):
    monkeypatch.setattr(llm, "FIND_JOB", None)
    monkeypatch.setattr(llm, "_chat", lambda *a: (_ for _ in ()).throw(llm.IncompleteReply("abort")))
    with pytest.raises(llm.IncompleteReply):
        llm.chat("s", "u")
