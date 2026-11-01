"""Step 2 of the search-model fine-tuning: teach an embedding model that a question and its table belong together.

Contrastive training with in-batch negatives (the "MultipleNegativesRankingLoss" idea, written out so every
step is visible): a batch holds B (question, table) pairs; every question is scored against all B tables in the
batch, and the loss pushes its own table to the top and the B-1 others down. Each batch has B different tables,
so no "negative" is secretly the right answer. The table side is the same text the retriever indexes
(retrieve.embedding_texts), so training and search see the same thing.

Works for bge-m3 (already a search model: learns Hagstofa's words) and for Miðeind's IceBERT (knows Icelandic
better, but is a fill-mask model: sentence-transformers adds mean pooling and it has to learn search from
scratch). Train both on the same questions and compare on dev:

    .venv/bin/python scripts/finetune_embed.py BAAI/bge-m3      --out models/bge-m3-hagstofa
    .venv/bin/python scripts/finetune_embed.py mideind/IceBERT  --out models/icebert-hagstofa --lr 3e-5
    EMBED_MODEL=models/bge-m3-hagstofa .venv/bin/python eval/run_eval.py -r all -q dev    # step 3: measure
    ... --max-steps 2 --batch 4          # smoke test (runs on CPU)

Needs a GPU for real runs (bge-m3 is 568M parameters: ~10 GB for weights, gradients and optimizer, plus
activations). On AMD (LUMI) the ROCm build of torch also reports itself as "cuda".
"""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import embedding_texts, load_catalog  # noqa: E402

QUESTIONS = ROOT / "data" / "finetune" / "train_questions.jsonl"
SEED = 20261030


def load_pairs():
    """[(question, table text)], with the table text exactly as the retriever embeds it."""
    tables = {}
    for t in load_catalog():
        tables.setdefault(t["id"], t)  # some ids are in two folders; train on the first, as indexed first
    ids = list(tables)
    texts, _ = embedding_texts([tables[i] for i in ids])
    text_of = dict(zip(ids, texts))
    rows = [json.loads(line) for line in open(QUESTIONS, encoding="utf-8")]
    return [(r["q"], r["table"], text_of[r["table"]]) for r in rows if r["table"] in text_of]


def batches(pairs, size, rng):
    """Shuffled batches in which every table appears at most once (its other questions wait for a later batch)."""
    by_table = defaultdict(list)
    for p in pairs:
        by_table[p[1]].append(p)
    for qs in by_table.values():
        rng.shuffle(qs)
    queues = list(by_table.values())
    while queues:
        rng.shuffle(queues)
        round_ = [q.pop() for q in queues]  # one question per table
        queues = [q for q in queues if q]
        for i in range(0, len(round_) - 1, size):  # a batch of 1 has no negatives: skip it
            yield round_[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="BAAI/bge-m3, mideind/IceBERT, mideind/IceBERT-large, or a local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="more = more negatives per question = usually better")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="similarity scale (inverse temperature)")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-steps", type=int, help="stop early (smoke test)")
    ap.add_argument("--checkpointing", action="store_true", help="trade speed for memory (bigger batches)")
    args = ap.parse_args()

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs()
    steps_per_epoch = math.ceil(len(pairs) / args.batch)
    total = min(args.epochs * steps_per_epoch, args.max_steps or 10**9)
    print(f"{len(pairs)} pairs, {len({p[1] for p in pairs})} tables, ~{total} steps, device {device}")

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_len
    if args.checkpointing:
        model[0].auto_model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, total // 10)  # linear warm-up, then linear decay
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))

    prepare = getattr(model, "preprocess", None) or model.tokenize  # tokenize() is deprecated in sentence-transformers 6

    def embed(texts):
        features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prepare(texts).items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(features)["sentence_embedding"]
        return F.normalize(out.float(), dim=-1)

    step, start, running = 0, time.time(), []
    for epoch in range(args.epochs):
        for batch in batches(pairs, args.batch, rng):
            q = embed([p[0] for p in batch])
            t = embed([p[2] for p in batch])
            scores = q @ t.T * args.scale  # [B, B]: question i against every table in the batch
            loss = F.cross_entropy(scores, torch.arange(len(batch), device=device))  # right table = diagonal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            running.append(loss.item())
            if step % 20 == 0 or step == total:
                print(f"  epoch {epoch + 1} step {step}/{total}  loss {sum(running) / len(running):.3f}"
                      f"  ({time.time() - start:.0f} s)")
                running = []
            if step >= total:
                break
        if step >= total:
            break

    model.eval()
    model.save(args.out)
    print(f"saved {args.out}; measure it: EMBED_MODEL={args.out} .venv/bin/python eval/run_eval.py -r all -q dev")


if __name__ == "__main__":
    main()
