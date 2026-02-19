import argparse
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipImageProcessor, AutoTokenizer
from PIL import Image
from pathlib import Path
import torch
import os

class Roii:
    """
    Roii: ROI Inference from images using BLIP2 captions.
    Maps visual content to likely fMRI-activated cortical regions.
    """

    ROI_MAP = {
        "face": "Fusiform Face Area (FFA)",
        "person": "Fusiform Face Area (FFA)",
        "people": "Fusiform Face Area (FFA)",
        "body": "Extrastriate Body Area (EBA)",
        "place": "Parahippocampal Place Area (PPA)",
        "scene": "Parahippocampal Place Area (PPA)",
        "word": "Visual Word Form Area (VWFA)",
        "text": "Visual Word Form Area (VWFA)"
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = '''
        This image maximizes a fMRI activation certain brain cortex. 
        Infer the function of the cortex based of the visual of the image.
        "face": "Fusiform Face Area (FFA)",
        "person": "Fusiform Face Area (FFA)",
        "people": "Fusiform Face Area (FFA)",
        "body": "Extrastriate Body Area (EBA)",
        "place": "Parahippocampal Place Area (PPA)",
        "scene": "Parahippocampal Place Area (PPA)",
        "word": "Visual Word Form Area (VWFA)",
        "text": "Visual Word Form Area (VWFA)"
        '''
        model_name = "Salesforce/blip2-flan-t5-xl"
        hf_cache = "./hf_cache"
        
        os.makedirs(hf_cache, exist_ok=True)

        # Load tokenizer and processor
        print("Load tokenizer and processor")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache, use_fast=False)
        image_processor = BlipImageProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        self.processor = Blip2Processor(image_processor=image_processor, tokenizer=self.tokenizer)

        # Load model
        print("Load model")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def caption_image(self, image_path, max_new_tokens=1000):
        """Generate a BLIP2 caption for a single image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def infer_roi(self, caption):
        """Map caption keywords to likely brain cortex regions."""
        caption_lower = caption.lower()
        for keyword, roi in self.ROI_MAP.items():
            if keyword in caption_lower:
                return roi
        return "Unknown"

    def process_image(self, image_path, max_new_tokens=50):
        """Process a single image: caption + ROI inference."""
        caption = self.caption_image(image_path, max_new_tokens=max_new_tokens)
        roi = self.infer_roi(caption)
        return {"filename": Path(image_path).name, "caption": caption, "roi": roi}

    def process_folder(self, image_folder, max_new_tokens=50, extensions=(".png", ".jpg", ".jpeg", ".bmp"), N = None):
        """Process all images in a folder."""
        print(f"Start processing {image_folder}")
        results = []
        for filename in os.listdir(image_folder):
            print(f"Start processing {filename}")
            if filename.lower().endswith(extensions):
                image_path = os.path.join(image_folder, filename)
                result = self.process_image(image_path, max_new_tokens=max_new_tokens)
                results.append(result)
                print(f"{filename} -> ROI: {result['roi']}, Caption: {result['caption']}")
            if not N is None:
                if len(results) == N:
                    return results
        return results

