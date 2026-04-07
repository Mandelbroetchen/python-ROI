from roid.imset import Imset
from roit.roit import Roit

from pathlib import Path
import numpy as np
import torch

roit = Roit()

imset = Imset(Path("./datasets/tiny-test-clips-5"))


ref = torch.tensor(
    imset["tiny-test-5-EBA-True-0d7-0d6-42"]["test_0.JPEG.json"]["reference"],
    dtype=torch.float32)
emb = torch.tensor(
    imset["tiny-test-5-EBA-True-0d7-0d6-42"]["test_0.JPEG.json"]["embedded"], 
    dtype=torch.float32)
dif = emb - ref

from clip_interrogator import Config, Interrogator
import torch

ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))

clip_tensor = dif  # shape: (1024,)
clip_tensor = clip_tensor / clip_tensor.norm(dim=-1, keepdim=True)

prompt = ", ".join([
    *ci.rank_top(clip_tensor, ci.mediums.labels   ),
    *ci.rank_top(clip_tensor, ci.movements.labels),
    *ci.rank_top(clip_tensor, ci.flavors.labels),
    *ci.rank_top(clip_tensor, ci.artists.labels),
])

print(prompt)