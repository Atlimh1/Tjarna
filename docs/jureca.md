# Running on JURECA (or any Slurm cluster with GPUs)

How we ran the solution for the demo: the language models as GPU jobs, the pipeline and the web page on a login node,
and the presenting laptop connected by an SSH tunnel.

```
laptop browser ── http://127.0.0.1:8765 ── SSH tunnel ──► login node: main.py (search, data, checks, the page)
                                                                │
                                                                └──► GPU nodes: Gemma 4 31B, Qwen3-30B (vLLM)
```

## 1. The models: GPU jobs

`scripts/serve-gemma.sbatch` and `scripts/serve-qwen.sbatch` serve each model with vLLM on one node with 4 GPUs, as an
OpenAI-compatible API on port 8000. Both expect a project folder `$W` with a Python env holding vllm, the model
weights, a Hugging Face cache (compute nodes have no internet) and an API key file (see the scripts' headers).

```bash
W=<project folder> sbatch -A <project> scripts/serve-gemma.sbatch   # the model of every answer
W=<project folder> sbatch -A <project> scripts/serve-qwen.sbatch    # optional: the second opinion
squeue -u $USER                                                      # wait for RUNNING
grep "Application startup complete" serve-gemma-<job>.log            # loaded after 5-10 minutes
```

To have them ready at a given time, book ahead: `sbatch --begin=2026-11-01T20:15:00 ...`. JURECA's `dc-gpu-devel`
partition allows 2 running jobs per user, 2 hours each, and 4 submitted jobs.

## 2. The server: a login node

```bash
cd <repo> && git pull
source <venv>/bin/activate                       # Python 3.12+ with requirements.txt installed
G=$(squeue -u $USER -h -n gemma-vllm -t RUNNING -o %N | head -1)
Q=$(squeue -u $USER -h -n qwen-vllm -t RUNNING -o %N | head -1)
KEY=$(cat <project folder>/vllm.key)
export LLM_BASE_URL=http://${G}i:8000/v1 LLM_MODEL=gemma LLM_API_KEY=$KEY LLM_FIND_JOB=gemma-vllm \
       HF_HOME=<hf cache> HF_HUB_OFFLINE=1 GAGNVIST_STARTER=<gagnvist-starter with its data unpacked> \
       EMBED_MODEL_2=<folder of the fine-tuned bge-m3>        # optional: the fourth search
[ -n "$Q" ] && export LLM2_BASE_URL=http://${Q}i:8000/v1 LLM2_MODEL=qwen3-30b LLM2_API_KEY=$KEY
uvicorn main:app --host 127.0.0.1 --port 8765    # ~1 min to load the search models
```

- `<node>i` is the node's internal address (the `i` matters on JURECA).
- `-t RUNNING | head -1` takes one running job: two Gemma jobs at once would otherwise put two names in the URL.
- `LLM_FIND_JOB=gemma-vllm`: when that Gemma job ends, the server moves to another running `gemma-vllm` job by itself
  and asks again; without it, a new job means restarting the server.
- The second-opinion button appears only when `LLM2_*` is set and that model answers.
- The server reads the page (`web/index.html`) on every visit; Python changes need a restart.

## 3. The laptop: an SSH tunnel

The server listens only inside the login node, so the laptop forwards a local port to it:

```bash
ssh -i <key> -N -L 8765:127.0.0.1:8765 <user>@jureca<NN>.fz-juelich.de
```

`<NN>` is the login node the server runs on (`jrlogin12` in the server's prompt is `jureca12.fz-juelich.de` from
outside). After the one-time code the terminal shows nothing more: the tunnel is open. Then open:

- http://127.0.0.1:8765/health first: `"status": "ok"` means the model answers too
- http://127.0.0.1:8765/?view=demo for the presentation, or `/` and `/?debug=1` (see the README)
- http://127.0.0.1:8765/gradio, the chat, as a fallback

Several laptops (or people) can open the same tunnel at once; each browser keeps its own conversation.
Ask one question before presenting: the first answer after a start is slower.

## If something fails

| Problem | Do |
|---|---|
| The page says the model isn't answering | `squeue -u $USER`: no Gemma job running? Book one (5-10 min to load). With `LLM_FIND_JOB` the server finds it by itself |
| `/health` says `llm_unreachable` | the same; the job may still be loading |
| The tunnel won't connect | check the login node's name (`jurecaNN`, not `jrloginNN`) and the key |
| The page misbehaves | `/gradio`: the chat, same server, same answers |
