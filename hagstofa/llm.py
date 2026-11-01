"""LLM access through any OpenAI-compatible endpoint, so the model is a config choice.

    Ollama (default):  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen3:4b-instruct
    vLLM on DGX/LUMI:  LLM_BASE_URL=http://<host>:8000/v1      LLM_MODEL=<served model name>
    Hosted APIs:       set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to the provider's OpenAI-compatible endpoint
    LLM_TIMEOUT:       seconds per call (default 1800). The openai client's own default is 600 s, which a big
                       model with CPU offload exceeds on long prompts; it then retries the same call twice.
    LLM_FIND_JOB:      on JURECA, the Slurm job name of the model server (gemma-vllm): the address is found in the queue,
                       and a model that goes away mid-demo (its job ended) is replaced by another running job of
                       that name, the call tried once more (find_model). Unset: LLM_BASE_URL as given, as before.
    LLM_REASONING:     passed as reasoning_effort ("none", "low", "medium", "high"); unset = the model's default.
                       gemma4 thinks before answering by default; "none" made a short reply 2 s instead of 20 s.
"""
import contextlib
import contextvars
import json
import logging
import os
import subprocess
import threading
import time

import openai
from openai import OpenAI

_client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "none"),
    timeout=float(os.environ.get("LLM_TIMEOUT", 1800)),
)
MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
FIND_JOB = os.environ.get("LLM_FIND_JOB")
_switch = threading.Lock()
REASONING = os.environ.get("LLM_REASONING") or None


_override = contextvars.ContextVar("llm_override", default=None)
TRACE = contextvars.ContextVar("llm_trace", default=None)  # a list: every call is recorded there (the developer view)
NAMES = {"gemma": "Gemma 4 31B", "qwen3-30b": "Qwen3-30B-A3B", "qwen72b": "Qwen2.5-72B", "qwen2.5-7b": "Qwen2.5-7B",
         "gptoss": "gpt-oss-120b"}  # served name -> what the user sees under an answer


@contextlib.contextmanager
def using(base_url, model, api_key="none"):
    """Every chat() inside the block, in this thread, goes to another model: the second opinion."""
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=_client.timeout)
    token = _override.set((client, model))
    try:
        yield
    finally:
        _override.reset(token)


def current():
    """(client, model) for chat() right here: the override if one is active, else the configured model."""
    return _override.get() or (_client, MODEL)


def display_name(model=None):
    model = model or current()[1]
    return NAMES.get(model, model)


def chat(system, user, json_mode=False, temperature=0.0):
    try:
        return _chat(system, user, json_mode, temperature)
    except UNREACHABLE:
        # the model went away (its JURECA job ended): with LLM_FIND_JOB, move to another running job, ask once more
        if not (FIND_JOB and _override.get() is None and find_model()):
            raise
        return _chat(system, user, json_mode, temperature)


def _chat(system, user, json_mode, temperature):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    if REASONING:
        kw["reasoning_effort"] = REASONING
    client, model = current()
    started = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kw,
    )
    content = r.choices[0].message.content
    if getattr(r.choices[0], "finish_reason", None) not in (None, "stop", "tool_calls"):
        # 1 Nov, 14:21: the GPU job stopped mid-reply and "Árið 2" was shown as an answer, checked: a reply that
        # didn't end normally ("abort", "length") is an error, never an answer
        raise IncompleteReply(f"the model stopped mid-reply ({r.choices[0].finish_reason})")
    if (trace := TRACE.get()) is not None:
        trace.append({"system": system, "seconds": time.perf_counter() - started, "reply": content, "at": started,
                      "model": display_name(model)})
    return content


class IncompleteReply(RuntimeError):
    """The model's reply was cut off."""


# the model server can't be reached or stopped answering: the user is told so, not given a generic error
UNREACHABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, IncompleteReply)


def reachable(timeout=3.0):
    """Whether the model server answers right now (its model list, one quick request, no retries); with
    LLM_FIND_JOB, another running job is looked for first when it doesn't."""
    client, _ = current()
    try:
        client.with_options(timeout=timeout, max_retries=0).models.list()
        return True
    except Exception:  # noqa: BLE001 - any failure means: not reachable
        return bool(FIND_JOB and _override.get() is None and find_model())


def running_nodes(job=None):
    """The nodes of this user's RUNNING Slurm jobs named `job`, the one that ends last first; [] off a cluster."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "RUNNING", "-n", job or FIND_JOB,
                              "-o", "%e %N"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [node for _, node in sorted((line.split() for line in out.splitlines() if len(line.split()) == 2), reverse=True)]


def find_model(port=8000):
    """Point the client at a running model job that answers (LLM_FIND_JOB): its node's internal address
    (<node>i:<port>, as in docs/DEMO.md). True when one was found."""
    global _client
    with _switch:  # several questions may fail at once: one looks, the others use what it found
        for node in running_nodes():
            candidate = _client.with_options(base_url=f"http://{node}i:{port}/v1")
            try:
                candidate.with_options(timeout=5.0, max_retries=0).models.list()
            except Exception:  # noqa: BLE001 - loading or gone: try the next
                continue
            if str(candidate.base_url) != str(_client.base_url):
                logging.warning("model moved to %s (job %s)", candidate.base_url, FIND_JOB)
            _client = candidate
            return True
        return False


def parse_json(text):
    """The first JSON object in text, ignoring prose or fences around it and anything after it
    (qwen3:4b sometimes returns two objects in a row: json.loads fails with "Extra data")."""
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object", text, 0)
    return json.JSONDecoder().raw_decode(text, start)[0]


def chat_json(system, user):
    """chat() that returns a parsed dict, tolerating models that wrap JSON in prose or fences."""
    return parse_json(chat(system, user, json_mode=True))
