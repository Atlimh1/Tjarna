# Development notes

Setup details, choosing the model, and the evaluation tools. The README is the overview.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install requests rank_bm25 scikit-learn openai gradio pytest islenska tokenizer
.venv/bin/python scripts/crawl_catalog.py   # ~0.5 table/s (1 req/s), resumable; writes data/catalog.jsonl
.venv/bin/python -m pytest -q tests
.venv/bin/python app.py                     # http://localhost:7860
```

## Numbers from the organizers' data release (no PX-Web needed)

`px.query` reads each table's `data.px` from the organizers' release (snapshot of 18 Oct 2026, the same numbers as
their answer key) when the table is there, and asks PX-Web only otherwise. No rate limits (PX-Web blocked us with
429 for over an hour on 30 Oct), and fast. Needs gagnvist-starter next to this repo (or `GAGNVIST_STARTER`) and its
data unpacked (`GAGNVIST_DATA`, default `<starter>/data`):

```bash
cd ../gagnvist-starter && cat data-release/*.zip.part* > /tmp/d.zip && python3 -c "import zipfile; zipfile.ZipFile('/tmp/d.zip').extractall('.')" && rm /tmp/d.zip
```
(both languages, 5.4 GB; or `gagnvist get-data`, which also checks every file). `PX_SOURCE=live` turns it off.

**Editions and fallbacks** (`pipeline.fetch_part`, `answer`). Most tables have an Icelandic and an English edition, with
their own codes and sometimes their own numbers; the organizers' answer key reads English questions from the English
edition. Per sub-question the pipeline tries, until one gives at least one number (not all ".."):
tables (the picked one, then the next by search rank, `MAX_TABLES`) x editions (the question's language first, then
the other) x sources (the data release, then PX-Web). What was skipped is in each part's `fallback` (API: `frávik`).
Nothing anywhere: the part is "no data" and the answer says so; it never guesses a number.

## Server: JSON API + web page + chat (main.py)

```bash
.venv/bin/uvicorn main:app --port 8000      # page at http://127.0.0.1:8000, chat at /gradio, API docs at /docs (~1 min to load bge-m3)
```

| Endpoint | Request | Response |
|---|---|---|
| `POST /spyrja_px_api` | `{"spurning": "..."}` (or `"prompt"`, as in the Track 2 README) | the Track 2 starter's `svar`, `töflu_slóð`, `notud_sia`, plus `heimildir` (per table: id, title, link, updated, values chosen, rows read, calculation), `sekúndur`, `villa` |
| `POST /spyrja` | `{"spurningar": ["...", "..."]}` (1–10) | `{"svör": [...]}`: the same objects, in order, answered in parallel |
| `POST /annad_alit` | `{"spurning": "...", "fyrra_svar": "..."}` | second opinion: `svar` = the same object from the second model (LLM2_*), `sama` (same number or not), `samanburður` (one line); `tengt: false` when no second model is set |
| `GET /health` | | status, LLM model, whether a second model is set |
| `GET /` | | the web page (`web/index.html`): one HTML file that uses only the API above; the Gradio chat is at `/gradio` |

All answers share one pool of `MAX_PARALLEL` threads (default 4): the LLM server is the bottleneck. An error in one
question is reported in its `villa` and doesn't fail the rest. The API has no login: it binds to 127.0.0.1 by default;
`APP_AUTH=user:pass` protects only the Gradio chat.

Fields the web page reads, added next to the starter's (which are unchanged): per table `val_merkt` (values chosen, by
label), `forsendur` (values code filled in), `leit` (the search's top 5, the one used marked), `undirspurning`,
`titill_en`; per answer `svar_án_forsendna` (the answer without its "Miðað við" line, shown apart), `líkan`, `vissa`
(the confidence line), `vissustig`, `tölur_athugaðar` (the number check passed), `útreikningur_milli_taflna`,
`athugun_líkans` (the LLM topic check ran), `leit_án_töflu` (the search's top 5 when no table had numbers). The page
computes nothing: it lays these out and writes numbers with the reader's separators.

## GitHub Codespaces

Open the repo in a Codespace (Code → Codespaces → Create). `.devcontainer/devcontainer.json` builds it:
Python 3.12, CPU-only torch, `requirements.txt`, the Icegrams model, then runs the tests (~5–10 min the
first time). The catalog and bge-m3 table vectors are in `data/`, so retrieval and the eval work right away;
bge-m3 itself (~2.2 GB) downloads on first use of `embed`/`fusion`. Port 8000 (the server) is
forwarded; Codespaces adds its own login.

```bash
.venv/bin/python eval/run_eval.py -r all -q dev
.venv/bin/uvicorn main:app --port 8000
```

## Choosing the LLM

Any OpenAI-compatible endpoint, set by env vars (see `hagstofa/llm.py`):

| Where | LLM_BASE_URL | LLM_MODEL |
|---|---|---|
| Ollama (default) | `http://localhost:11434/v1` | `qwen3:4b-instruct` |
| vLLM on DGX Spark / LUMI | `http://<host>:8000/v1` | served model name |
| Hosted API (if rules allow) | provider's OpenAI-compatible URL + `LLM_API_KEY` | provider model id |

Finding so far: a 4B model on a Pi gets the plumbing right but picks wrong variable values and writes
poor Icelandic. Plan for a 27B–70B class model on the DGX.

## Evaluation: run after every change

```bash
.venv/bin/python eval/run_eval.py          # retrieval hit rate, no LLM needed
.venv/bin/python eval/run_eval.py --e2e -q eval/questions_e2e.jsonl   # table picked + correct number
```

The organizers' scoring (`gagnvist evaluate`, their 60 questions) on our pipeline, with their code unchanged.
Needs a clone of gagnvist-starter (`--starter PATH`, `GAGNVIST_STARTER`, or `../gagnvist-starter`) and the LLM:

```bash
git clone https://github.com/SyedMuhammadAzhar/gagnvist-starter ../gagnvist-starter
.venv/bin/python eval/gagnvist_eval.py                     # all 60; --language is, --level easy, --only reykjavik
```

Ablation ladder: the same 60 questions answered by the pipeline built up one step at a time (LLM alone, bge-m3
search, our search, + LLM picks the table, + calculation, + two-table split, + English), one table row per step:

```bash
LLM_MODEL=gemma4:26b LLM_REASONING=none .venv/bin/python eval/ladder.py     # ~2 h with Ollama on a small GPU
.venv/bin/python eval/ladder.py --rungs 0,2,6 --limit 6                       # a quick look
```

### Fine-tuning the search model (bge-m3 vs Miðeind's IceBERT)

The LLM itself isn't fine-tuned: numbers must come live from PX-Web, and given the right table gemma4:26b
already picks the right values every time. Retrieval is the bottleneck, so the embedding model is what we train,
on question → table pairs for the ~950 current tables that are in no eval set (dev stays a fair test).

```bash
# 1. training questions (needs a fast LLM; resumable; try --limit 20 and read them first)
LLM_BASE_URL=... LLM_MODEL=... LLM_REASONING=none .venv/bin/python scripts/finetune_questions.py
# 2. train (GPU), both models on the same questions
.venv/bin/python scripts/finetune_embed.py BAAI/bge-m3     --out models/bge-m3-hagstofa
.venv/bin/python scripts/finetune_embed.py mideind/IceBERT --out models/icebert-hagstofa --lr 3e-5
# 3. measure on dev (table vectors are computed on first use: minutes on a GPU)
EMBED_MODEL=models/bge-m3-hagstofa .venv/bin/python eval/run_eval.py -r all -q dev
```
Keep a model only if it beats the current embed/checked numbers on dev beyond noise (±3 points), then confirm once on test.

### Comparing LLMs

`eval/questions_e2e.jsonl` has 24 questions with `truth` (built by `eval/build_e2e.py` from value texts, so
add questions there, not by hand). `eval/compare_models.py` runs them through each model three ways:
`pick` (choose among the top-8), `oracle` (given the right table: right values? right number?) and `full`
(the whole pipeline). Every answer is saved to `eval/results/` for reading the Icelandic.

```bash
LLM_BASE_URL=http://<gpu-machine>:11434/v1 .venv/bin/python eval/compare_models.py gemma4:26b gpt-oss:20b
.venv/bin/python eval/compare_models.py gemma4:e2b --modes pick      # no PX-Web calls
```

`eval/questions.jsonl`: `{"q", "table", "truth": {var_code: [value_code]}}`. The expected number is
fetched live using `truth`, so the set stays valid as Hagstofa publishes new data. Add questions
whenever you find a failure.

## The first pipeline (Oct 2026)


```
question ─► retrieve.py ──► top-k candidate tables (char n-gram TF-IDF over data/catalog.jsonl)
         ─► pipeline.pick_table ──► LLM picks one table
         ─► pipeline (select)   ──► LLM picks variable value codes; validate_selections() drops
                                    invented codes and fills defaults (latest period, "Alls")
         ─► px.query            ──► live numbers as json-stat2 → flatten() → rows
         ─► pipeline (answer)   ──► LLM writes answer from rows only; app.py appends citation + link
```

The citation (table id, title, link, last-updated) is added by code, not the LLM, so it can't be hallucinated.
