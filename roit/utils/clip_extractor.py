import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.utils import data
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from brainactiv.dataset.nsd import NaturalScenesDataset

class CLIPExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        )

    def extract_for_dataset(self, dataset: "NaturalScenesDataset"):
        assert dataset.partition == "all"
        assert dataset.transform is None
        features = []

        # Batch processing: you could define a batch size
        batch_size = 16
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_imgs = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
            x = self.forward(batch_imgs).detach().cpu().numpy()  # batch embeddings
            features.append(x)

        features = np.concatenate(features, axis=0).astype(np.float32)
        return features

    @torch.no_grad()
    def forward(self, imgs):
        """
        imgs: either a single PIL.Image or a list of PIL.Images
        returns: torch.Tensor of shape (batch_size, embedding_dim)
        """
        # Wrap single image into list
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        # Preprocess the batch
        inputs = self.processor(
            images=imgs, return_tensors="pt", padding=True
        )['pixel_values'].to(self.device)  # shape: (batch_size, 3, H, W)

        # Forward through CLIP
        outputs = self.clip(inputs).image_embeds  # (batch_size, embedding_dim)
        return outputs
