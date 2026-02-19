# Appendix

## A. System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for optimal performance)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: Sufficient space for input images and transformed outputs

### Software Requirements
- Python 3.8+
- PyTorch with CUDA support
- Required Python packages (see requirements.txt):
  ```
  torch
  torchvision
  diffusers
  transformers
  pillow
  numpy
  matplotlib
  python-dotenv
  ip-adapter
  ```

## B. Installation Guide

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file in the project root with the following variables:
   ```
   IPADAPTER=/path/to/ip_adapter_models
   CHECKPOINTS=/path/to/roi_checkpoints
   MODEMBED=/path/to/modulation_embeddings
   ```

4. **Download required models**:
   - Stable Diffusion v1.5
   - CLIP-ViT-H-14-laion2B-s32B-b79K
   - ROI-specific checkpoints
   - IP-Adapter models

## C. Advanced Configuration

### Custom ROI Definitions

To add custom ROIs:

1. Extend the `ROI` list in `roit.py`:
   ```python
   ROI = ["FFA", "EBA", ..., "CUSTOM_ROI"]
   ```

2. Add corresponding model checkpoints in the `CHECKPOINTS` directory:
   ```
   subj1_CUSTOM_ROI.ckpt
   ```

3. Add modulation embeddings in the `MODEMBED` directory:
   ```
   subj1_CUSTOM_ROI_mod_embed_max.npy
   subj1_CUSTOM_ROI_mod_embed_min.npy
   ```

### Performance Optimization

1. **Batch Processing**:
   - Adjust batch size in `CLIPExtractor.extract_for_dataset()`
   - Larger batches improve throughput but require more memory

2. **Device Management**:
   - The system automatically uses CUDA if available
   - For multi-GPU setups, modify the device selection logic

3. **Caching**:
   - The `Imset` class implements singleton pattern for path-based caching
   - The `Roit` class caches modulation embeddings

## D. Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Process smaller image sets
   - Use lower resolution images

2. **Missing Environment Variables**:
   - Verify `.env` file exists in project root
   - Check variable names match those in `roit.py`
   - Ensure paths are absolute

3. **Image Loading Errors**:
   - Verify supported image formats
   - Check file permissions
   - Validate image integrity

4. **Model Loading Failures**:
   - Verify model files exist at specified paths
   - Check internet connection for downloading models
   - Ensure sufficient disk space

## E. Code Examples

### Custom Transformation Pipeline

```python
from roid.imset import Imset
from roit.roit import Roit

class CustomRoit(Roit):
    def __init__(self, custom_roi, **kwargs):
        super().__init__(**kwargs)
        self.roi = custom_roi

    def custom_transform(self, image):
        # Implement custom transformation logic
        embeds = self.modulated_embedding(image)
        # Add custom processing here
        return self.ip_model.generate(
            clip_image_embeds=embeds,
            image=image,
            strength=self.gamma,
            num_samples=1,
            num_inference_steps=50,
            seed=self.seed
        )[0]

# Usage
custom_roit = CustomRoit(custom_roi="MY_ROI", maximize=True, alpha=0.5)
imset = Imset("path/to/images")
transformed = custom_roit.transform_imset(imset)
transformed.save("path/to/output")
```

### Batch Processing with Custom Parameters

```python
from roid.roid import Roid
from roit.roit import Roit

# Define parameter combinations
params = [
    {"roi": "FFA", "maximize": True, "alpha": 0.7, "gamma": 0.6},
    {"roi": "EBA", "maximize": False, "alpha": 0.5, "gamma": 0.8},
    # Add more parameter combinations as needed
]

source_path = "path/to/source"
for param_set in params:
    roit = Roit(**param_set)
    roid = Roid(source_path, roit=roit)
    roid.transform()
    roid.target.save()
```

## F. Glossary

| Term | Definition |
|------|------------|
| ROI | Region of Interest - specific areas of the brain associated with particular visual processing functions |
| CLIP | Contrastive Language-Image Pre-Training - a neural network trained on image-text pairs |
| IP-Adapter | Image Prompt Adapter - a module that enables image-based conditioning in diffusion models |
| SLERP | Spherical Linear Interpolation - a method for interpolating between vectors on a sphere |
| Imset | Image Set - the custom data structure used to manage collections of images |
| Roid | ROI Dataset - the module for managing and processing image datasets |
| Roit | ROI Transform - the module for applying ROI-specific transformations to images |