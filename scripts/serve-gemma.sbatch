#!/bin/bash
# The language model behind every answer: Gemma 4 31B served by vLLM (OpenAI-compatible API) on one node, 4 GPUs.
#     W=<your project folder> sbatch -A <your project> scripts/serve-gemma.sbatch
# $W holds vllm-env/ (a Python env with vllm), hf/ (Hugging Face cache), models/gemma/ (the Gemma 4 31B weights)
# and vllm.key (the API key, any random string; the pipeline sends it as LLM_API_KEY). Compute nodes have no internet.
#SBATCH --job-name=gemma-vllm
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=serve-gemma-%j.log
W=${W:?set W to your project folder}
module load Python/3.13.5
source $W/vllm-env/bin/activate
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
echo "NODE: $(hostname)"   # the pipeline reaches it at http://<node>i:8000/v1 from a login node
vllm serve $W/models/gemma --served-model-name gemma \
  --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image":0}' --host 0.0.0.0 --port 8000 --api-key "$(cat $W/vllm.key)"
