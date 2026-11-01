"""Minimal client for Hagstofa Íslands' PX-Web API (v1)."""
import logging
import os
import threading
import time

import requests

from . import local_data

log = logging.getLogger(__name__)
SOURCE = os.environ.get("PX_SOURCE", "auto")  # auto: the data release when it has the table, else PX-Web; live

BASES = {
    "is": "https://px.hagstofa.is/pxis/api/v1/is",
    "en": "https://px.hagstofa.is/pxen/api/v1/en",
}
WEB_BASES = {
    "is": "https://px.hagstofa.is/pxis/pxweb/is",
    "en": "https://px.hagstofa.is/pxen/pxweb/en",
}
MIN_INTERVAL = 0.4  # seconds between calls; PX-Web servers commonly allow ~30 calls / 10 s

_session = requests.Session()
_last_call = 0.0
_pace = threading.Lock()  # the server answers questions in parallel threads; they share the pacing


def _wait_turn():
    """Sleep until MIN_INTERVAL has passed since the last call by any thread, then claim the slot."""
    global _last_call
    with _pace:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _request(method, url, attempts=5, **kw):
    """Throttled request; "busy" answers (429/503) and network errors (timeout, dropped connection)
    are retried with growing waits (5, 10, 20, 40 s), the last failure is raised."""
    for attempt in range(attempts):
        _wait_turn()
        last = attempt == attempts - 1
        try:
            r = _session.request(method, url, timeout=60, **kw)
        except (requests.ConnectionError, requests.Timeout):
            if last:
                raise
            time.sleep(2 ** attempt * 5)
            continue
        if r.status_code in (429, 503) and not last:
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.json()


def list_node(path="", lang="is"):
    """Children of a folder: [{id, type: 'l'|'t', text, ...}]."""
    return _request("GET", f"{BASES[lang]}/{path}".rstrip("/"))


def metadata(path, lang="is"):
    """Table metadata: {title, variables: [{code, text, values, valueTexts, time?, elimination?}]}."""
    return _request("GET", f"{BASES[lang]}/{path}")


def query(path, selections, lang="is"):
    """Fetch data as json-stat2. selections: {var_code: [value_code, ...]}.
    `path` is our catalog's (Icelandic) path; lang="en" reads the table's English edition, with its own codes.
    Reads the organizers' data release when it holds the table (see local_data.py), else PX-Web;
    PX_SOURCE=live always asks PX-Web."""
    if SOURCE != "live" and local_data.has(path, lang):
        try:
            return local_data.query(path, selections, lang)
        except Exception:  # noqa: BLE001 - a code the snapshot lacks, a broken file: PX-Web may still have it
            log.warning("local data failed for %s (%s), asking PX-Web", path, lang, exc_info=True)
    url, body = api_request(path, selections, lang)
    return _request("POST", url, json=body)


def api_request(path, selections, lang="is"):
    """(url, json body) of the PX-Web API query for these selections: what query() sends, and what the chat shows
    under "reproduce this answer" so anyone gets the same number from Statistics Iceland without us."""
    body = {
        "query": [
            {"code": code, "selection": {"filter": "item", "values": values}}
            for code, values in selections.items()
        ],
        "response": {"format": "json-stat2"},
    }
    return f"{BASES[lang]}/{local_data.api_path(path, lang)}", body


def web_url(path, lang="is"):
    """Human-facing PX-Web page for a table, for citations.
    API path  Ibuar/mannfjoldi/1_yfirlit/X/MAN00101.px  becomes
    web path  Ibuar/Ibuar__mannfjoldi__1_yfirlit__X/MAN00101.px"""
    *folders, table = local_data.api_path(path, lang).split("/")
    return f"{WEB_BASES[lang]}/{folders[0]}/{'__'.join(folders)}/{table}"
