"""Embed a JSON list of texts with a sentence-transformers model; runs on the GPU machine.

    venv/bin/python encode.py texts.json BAAI/bge-m3 out.npy
"""
import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

texts_file, model_name, out = sys.argv[1:4]
texts = json.load(open(texts_file, encoding="utf-8"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
if device == "cuda":
    model.half()  # fp16: half the memory, so bge-m3 fits easily in 4 GB

start = time.time()
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
np.save(out, vectors.astype(np.float32))
print(f"{len(texts)} texts -> {vectors.shape} on {device} in {time.time() - start:.1f}s")
