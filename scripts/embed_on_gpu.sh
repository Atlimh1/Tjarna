#!/usr/bin/env bash
# Compute table embeddings on the GPU machine and put them in data/ where EmbeddingRetriever finds them.
#   scripts/embed_on_gpu.sh [model]        default model: BAAI/bge-m3
# Needs: SSH key access to $GPU_HOST and ~/hagstofan-gpu/venv there (torch + sentence-transformers).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-BAAI/bge-m3}"
# GPU machine: GPU_HOST=user@host
: "${GPU_HOST:?set GPU_HOST=user@host of a machine with ~/hagstofan-gpu/venv}"
REMOTE=hagstofan-gpu
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

CACHE=$(.venv/bin/python - "$MODEL" "$TMP/texts.json" <<'EOF'
import json, sys
sys.path.insert(0, ".")
from hagstofa.retrieve import embedding_texts, load_catalog
texts, cache = embedding_texts(load_catalog(), sys.argv[1])
json.dump(texts, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print(cache)
EOF
)
echo "model $MODEL -> $CACHE"
scp -q "$TMP/texts.json" scripts/encode.py "$GPU_HOST:$REMOTE/"
ssh "$GPU_HOST" "cd $REMOTE && venv/bin/python encode.py texts.json '$MODEL' out.npy"
scp -q "$GPU_HOST:$REMOTE/out.npy" "$CACHE"
echo "saved $CACHE"
