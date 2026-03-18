import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from ip_adapter import IPAdapter

from brainactiv.dataset.nsd_clip import CLIPExtractor
from brainactiv.methods.dino_encoder import EncoderModule, DINO_TRANSFORM
from brainactiv.methods.slerp import slerp

#from .utils.slerp import slerp
#from .utils.clip_extractor import CLIPExtractor
from .utils.log_time import log_time
from roid.imset import Imset

class Roit:

    ROI = ["FFA", "EBA", "VWFA", "OPA", "PPA", "RSC", "V1", "V2", "V3", "V4"]
    
    @log_time
    def __init__(
        self,
        roi="EBA", 
        maximize=True, 
        alpha=0.7,
        gamma=0.6,
        seed=42
    ):
        self.roi = roi
        self.maximize = maximize
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self._mod_embed = {}

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"""[Roit] Initialized with 
    roi = {self.roi}, 
    maximize = {self.maximize}
    alpha = {self.alpha}
    gamma = {self.gamma}
    seed = {self.seed}
    device = {self.device}""")

        # Load dotenv and device
        load_dotenv()
        self.IPADAPTER = Path(os.getenv("IPADAPTER"))
        self.CHECKPOINTS = Path(os.getenv("CHECKPOINTS"))
        self.MODEMBED = Path(os.getenv("MODEMBED"))

        print(f"""[Roit] Load dotenv:
    IPADAPTER = {self.IPADAPTER}
    CHECKPOINTS = {self.CHECKPOINTS}
    MODEMBED = {self.MODEMBED}""")

        # Load models
        print("[Roit] Loading models...")
        self._load_diffusion_pipeline()
        self._load_ip_adapter()
        self._load_clip()
        #self._load_dino_encoder()
        print("[Roit] All models loaded successfully")

    #@log_time    
    def _load_diffusion_pipeline(self):
        self.diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(self.device)
        self.diffusion_pipeline.scheduler = DDIMScheduler.from_config(
            self.diffusion_pipeline.scheduler.config
        )
        self.diffusion_pipeline.safety_checker = None

    #@log_time
    def _load_ip_adapter(self):
        clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.ip_model = IPAdapter(
            self.diffusion_pipeline, 
            clip_model_name, 
            self.IPADAPTER, 
            self.device
        )

    #@log_time
    def _load_clip(self):
        self.clip_extractor = CLIPExtractor(self.device)
    
    #@log_time
    def _load_dino_encoder(self):
        self.dino_encoder = {}
        for roi in self.ROI:
            ckpt_path_dino = self.CHECKPOINTS / f"subj1_{roi}.ckpt"
            self.dino_encoder[roi] = EncoderModule.load_from_checkpoint(
                ckpt_path_dino, 
                strict=False
            ).to(self.device).eval()
    
    #@log_time
    def _load_mod_embed(self):
        name = f"subj1_{self.roi}_mod_embed_{'max' if self.maximize else 'min'}.npy"
        if name in self._mod_embed:
            return self._mod_embed[name]
        ckpt_path_embed = self.MODEMBED / name
        mod_embed = np.load(ckpt_path_embed)
        self._mod_embed[name] = mod_embed
        return mod_embed

    #@log_time
    def modulated_embedding(self, image_ref):
        image_ref_clip = self.clip_extractor(image_ref).detach().cpu().numpy()
        mod_embed = self._load_mod_embed()
        endpoint = mod_embed * np.linalg.norm(image_ref_clip)
        embeds = torch.from_numpy(
            slerp(image_ref_clip, endpoint, 1, t0=self.alpha, t1=self.alpha)
        ).unsqueeze(1).to(self.device)[0]
        return embeds
    
    #@log_time
    def transform(self, image_ref):
        with torch.no_grad():
            embeds = self.modulated_embedding(image_ref)
            image_new = self.ip_model.generate(
                clip_image_embeds=embeds,
                image=image_ref,
                strength=self.gamma,
                num_samples=1,
                num_inference_steps=50,
                seed=self.seed
            )[0]
        return image_new
    
    #@log_time
    def transform_imset(self, imset_ref):
        imset_new = Imset()

        trans = lambda x: str(x).replace(".", "d")
        suffix = f"{self.roi}-{self.maximize}-{trans(self.alpha)}-{trans(self.gamma)}-{self.seed}"
 
        imset_new.root = imset_ref.root.parent / f"{imset_ref.root.stem}-{suffix}{imset_ref.root.suffix}"
        for key, obj in imset_ref.items():
            print(f"[Roit] Transforming {key} in {imset_ref.root} to {self.roi}")
            if isinstance(obj, dict):
                imset_new[key] = self.transform_imset(obj)
            else:
                imset_new[key] = self.transform(obj)
        return imset_new