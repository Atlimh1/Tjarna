#!/bin/bash
# The second opinion: Qwen3-30B-A3B served by vLLM on one node, 4 GPUs (same folder layout as serve-gemma.sbatch).
#     W=<your project folder> sbatch -A <your project> scripts/serve-qwen.sbatch
#SBATCH --job-name=qwen-vllm
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=serve-qwen-%j.log
W=${W:?set W to your project folder}
module load Python/3.13.5
source $W/vllm-env/bin/activate
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
echo "NODE: $(hostname)"
vllm serve $W/models/Qwen3-30B-A3B-Instruct-2507 --served-model-name qwen3-30b \
  --tensor-parallel-size 4 --max-model-len 16384 --safetensors-load-strategy=prefetch \
  --host 0.0.0.0 --port 8000 --api-key "$(cat $W/vllm.key)"
