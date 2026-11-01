#!/bin/bash
# Search-model fine-tuning on JURECA, all steps in one job (one node, 4 x A100 40 GB), ~1 h:
#   1. vLLM serves Qwen3-30B on the node; scripts/finetune_questions.py writes training questions for the ~940
#      current tables that are in no eval set (dev/test/hand/muni/real/e2e, questions_new, the organizers' 60)
#   2. scripts/finetune_embed.py trains bge-m3 on (question -> table) into models/bge-m3-hagstofa-<time>
#   3. eval/run_eval.py on dev (all retrievers), real, muni and hand, with the base and the new model;
#      eval/experiments/gagnvist_retrieval.py on the organizers' 60 questions with the new model
# Nothing changes for the app: the new model is used only with EMBED_MODEL=<its folder>.
#
#   D=<your project folder> sbatch -A <your project> scripts/finetune_jureca.sbatch     (from the repo)
#   tail -f finetune-<job>.log
#SBATCH --job-name=finetune-search
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=finetune-%j.log
set -uo pipefail

D=${D:?set D to your project folder}            # vllm-env, models/, hf/ (Hugging Face cache), hagstofan/
REPO=$D/hagstofan
STAMP=$(date +%m%d-%H%M)
OUT=$REPO/models/bge-m3-hagstofa-$STAMP           # a new folder per run: the question-vector cache is keyed by it
module load Python/3.13.5
export HF_HOME=$D/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0   # compute nodes have no internet; no nvcc
export GAGNVIST_STARTER=$D/gagnvist-starter
echo "NODE $(hostname)  START $(date)  OUT $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- 1. training questions ---------------------------------------------------------------------------------
KEY=$(openssl rand -hex 16)
"$D/vllm-env/bin/vllm" serve "$D/models/Qwen3-30B-A3B-Instruct-2507" --served-model-name qwen3-30b \
    --tensor-parallel-size 4 --max-model-len 8192 --host 127.0.0.1 --port 8000 --api-key "$KEY" \
    > "$D/finetune-vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM=$!
trap 'kill $VLLM 2>/dev/null' EXIT
for i in $(seq 1 120); do  # up to 20 min to load
    curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models > /dev/null && break
    kill -0 $VLLM 2>/dev/null || { echo "vLLM died, see finetune-vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 10
done
echo "vLLM ready $(date)"
cd "$REPO" && source .venv/bin/activate
LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_API_KEY=$KEY LLM_MODEL=qwen3-30b \
    python scripts/finetune_questions.py --workers 32 || { echo "step 1 failed"; exit 1; }
echo "questions: $(wc -l < data/finetune/train_questions.jsonl)  $(date)"
kill $VLLM; wait $VLLM 2>/dev/null; trap - EXIT

# --- 2. train ------------------------------------------------------------------------------------------------
python scripts/finetune_embed.py BAAI/bge-m3 --out "$OUT" || { echo "step 2 failed"; exit 1; }
echo "trained $(date)"

# --- 3. measure: base vs new, same sets ----------------------------------------------------------------------
for MODEL in BAAI/bge-m3 "$OUT"; do
    echo; echo "===== EMBED_MODEL=$MODEL"
    EMBED_MODEL=$MODEL python eval/run_eval.py -r all -q dev 2>&1 | grep -E "top1"
    for Q in real muni hand; do
        f=$Q; [ "$Q" = hand ] || f=eval/questions_$Q.jsonl
        echo "-- $Q"; EMBED_MODEL=$MODEL python eval/run_eval.py -r checked -q "$f" 2>&1 | grep -E "^retrieval"
    done
done
echo; echo "===== organizers' 60 questions, new model"
EMBED_MODEL=$OUT python eval/experiments/gagnvist_retrieval.py "$GAGNVIST_STARTER" 2>&1 | grep -E "^==|top1"
echo "DONE $(date)  model: $OUT"
