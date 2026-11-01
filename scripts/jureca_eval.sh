#!/usr/bin/env bash
# The measurements that need the LLM, against our own Qwen (vLLM) job on JURECA. Run on a JURECA login node:
#
#     PROJECT=<folder holding each member's folder> bash scripts/jureca_eval.sh [owner of the vLLM job]   (default: you)
#
# From a clone without its own environment, reuse another member's (read access is enough):
#     VENV=$PROJECT/<member>/hagstofan/.venv \
#     GAGNVIST_STARTER=$PROJECT/<member>/gagnvist-starter \
#     HF_HOME=$PROJECT/<member>/hf bash scripts/jureca_eval.sh
#
# Finds the node of the running Qwen job (job name JOB_NAME, default qwen-vllm; other jobs such as bgem3-vllm may run
# too), points LLM_BASE_URL at it (login node -> compute node: node name + "i"), checks the model answers, then runs,
# most important first, each step even if an earlier one failed:
#   1-2. the organizers' 60 questions twice, pipeline only (before) and through the front door (after): same model,
#        same hour, so the difference is what the guard layers cost or gain
#   3.   our own frozen set (41 questions no rule was tuned on), through the front door
#   4.   the 16 questions Statistics Iceland can't answer: declined or not
#   5.   guard + LLM scope check        6. questions in other languages
# ~25-40 minutes.
set -euo pipefail
OWNER=${1:-$USER}
PROJECT=${PROJECT:?set PROJECT to the project folder (it holds <owner>/vllm.key)}
if [ -z "${LLM_BASE_URL:-}" ]; then  # another model: set LLM_BASE_URL and LLM_MODEL, e.g. http://jrc0384i:8001/v1 gemma
    NODE=$(squeue -u "$OWNER" -h -t RUNNING -n "${JOB_NAME:-qwen-vllm}" -o "%N" | head -1)
    if [ -z "$NODE" ]; then
        echo "No running ${JOB_NAME:-qwen-vllm} job for $OWNER: start it first (sbatch serve-qwen.sbatch), then wait for 'Application startup complete'."
        exit 1
    fi
    export LLM_BASE_URL="http://${NODE}i:8000/v1"
fi
export LLM_MODEL="${LLM_MODEL:-qwen3-30b}"
LLM_API_KEY="${LLM_API_KEY:-$(cat "$PROJECT/$OWNER/vllm.key")}"
export LLM_API_KEY
export HF_HOME="${HF_HOME:-$PROJECT/$USER/hf}"  # where bge-m3 is cached
if ! curl -sS -m 10 "$LLM_BASE_URL/models" -H "Authorization: Bearer $LLM_API_KEY" | grep -q "$LLM_MODEL"; then
    echo "The model doesn't answer at $LLM_BASE_URL yet (still starting? check the job's log)."
    exit 1
fi
echo "Qwen answers at $LLM_BASE_URL"

cd "$(dirname "$0")/.."
module load Python/3.13.5 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"
export GAGNVIST_STARTER="${GAGNVIST_STARTER:-$(pwd)/../gagnvist-starter}"  # the organizers' kit: data + evaluate()
export PYTHONDONTWRITEBYTECODE=1  # a shared .venv may not be writable
mkdir -p eval/results
LOG="eval/results/jureca-$LLM_MODEL-$(date +%Y%m%d-%H%M%S).log"
step() {  # a failing step is reported, and the next one still runs
    echo; echo "== $1 ($(date +%H:%M))"; shift
    "$@" || echo "!! step failed (exit $?): $*"
}
{
    echo "main $(git rev-parse --short HEAD), $LLM_BASE_URL"
    step "1. BEFORE: the organizers' 60 questions, pipeline only (no guard, no scope check, no number check)" \
        python eval/gagnvist_eval.py
    step "2. AFTER: the same 60 through the front door (guard, scope check, number check), as users get them" \
        python eval/gagnvist_eval.py --front-door
    step "3. Our own frozen set: 41 questions the rules were never tuned on, through the front door" \
        python eval/gagnvist_eval.py eval/questions_new.jsonl --front-door
    step "4. 16 questions Statistics Iceland can't answer (other agency, private, opinion, future): declined?" \
        python eval/decline_eval.py
    step "5. Guard + LLM scope check (100 real questions per set + all bad inputs)" \
        python eval/guard_eval.py --llm --limit 100
    step "6. Other languages (2 questions per language, compared with the English original)" \
        python eval/lang_eval.py --limit 2
} 2>&1 | tee "$LOG"
echo "Everything is in $LOG"
