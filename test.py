import os
import gc
from pathlib import Path

import torch
from clip_interrogator import Config, Interrogator

from roid.imset import Imset
from roit.roit import Roit

# ── Config ────────────────────────────────────────────────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATASET_PATH = Path("./datasets/tiny-test-clips-5")
SAMPLE_KEY   = "tiny-test-5-EBA-True-0d7-0d6-42"
IMAGE_KEY    = "test_0.JPEG.json"
CLIP_MODEL   = "ViT-H-14/laion2b_s32b_b79k"
BATCH_SIZE   = 16

# ── Load data ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

roit  = Roit()
imset = Imset(DATASET_PATH)

sample = imset[SAMPLE_KEY][IMAGE_KEY]

ref = torch.tensor(sample["reference"], dtype=torch.float32).to(device)
emb = torch.tensor(sample["embedded"],  dtype=torch.float32).to(device)
dif = emb - ref

# ── Load CLIP interrogator ────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()

config = Config(
    clip_model_name=CLIP_MODEL,
    clip_offload=True,
    caption_offload=True,
    chunk_size=512,
)
ci = Interrogator(config)

# ── Prepare query vector ──────────────────────────────────────────────────────

query = dif
if query.dim() == 1:
    query = query.unsqueeze(0)                    # [1, D]
query = query / query.norm(dim=-1, keepdim=True)  # L2-normalise

# ── Batched ranking ───────────────────────────────────────────────────────────

def rank_top_batched(ci, image_features, labels, top_count=1, batch_size=16):
    all_sims = []

    for i in range(0, len(labels), batch_size):
        batch  = labels[i : i + batch_size]
        tokens = ci.tokenize(batch).to(ci.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            text_features = ci.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ text_features.T).squeeze(0)
        all_sims.append(sims.cpu())

        del tokens, text_features, sims
        torch.cuda.empty_cache()

    all_sims    = torch.cat(all_sims)
    top_indices = all_sims.topk(top_count).indices
    return [labels[i] for i in top_indices.tolist()]

# ── Build prompt ──────────────────────────────────────────────────────────────

prompt_parts = []
for label_set in [ci.mediums, ci.movements, ci.flavors, ci.artists]:
    torch.cuda.empty_cache()
    gc.collect()
    tops = rank_top_batched(ci, query, label_set.labels, top_count=1, batch_size=BATCH_SIZE)
    prompt_parts.extend(tops)

prompt = ", ".join(prompt_parts)
print(prompt)