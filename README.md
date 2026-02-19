# ROI-Dataset and ROI-Transform Documentation

## Introduction

This documentation describes the ROI-Dataset (roid) and ROI-Transform (roit) systems, which together provide a framework for processing and transforming image datasets based on Region of Interest (ROI) specifications from neuroscience research. The system enables batch processing of image collections with specialized transformations that modulate image features according to brain activity patterns.

## System Overview

### ROI-Dataset (roid)

The ROI-Dataset module provides functionality for managing collections of images organized in hierarchical directory structures. It offers:

- Recursive image loading from directory structures
- Image resizing and format conversion
- Hierarchical organization preservation
- Batch processing capabilities

### ROI-Transform (roit)

The ROI-Transform module implements specialized image transformations based on:

- Stable Diffusion image-to-image pipelines
- IP-Adapter integration for image prompt conditioning
- Neuroscience-inspired ROI modulation
- CLIP-based feature extraction

## Core Components

### Imset Class

The `Imset` class serves as the primary data structure for managing image collections. Key features include:

1. **Singleton Pattern Implementation**:
   ```python
   _instances = {}
   def __new__(cls, path=None, resolution=(512, 512)):
       if path in cls._instances:
           return cls._instances[path]
   ```

2. **Hierarchical Organization**:
   - Subdirectories become nested `Imset` instances
   - Maintains original directory structure
   - Supports recursive operations

3. **Image Processing**:
   - Automatic resizing to specified resolution
   - RGB conversion for consistency
   - Support for multiple image formats (.png, .jpg, .jpeg, .bmp)

4. **Persistence**:
   ```python
   def save(self, path=None):
       """Save all images in the Imset to disk, preserving directory structure"""
   ```

### Roid Class

The `Roid` class orchestrates the transformation pipeline:

1. **Initialization**:
   ```python
   def __init__(self, source_path, target_path=None, roit=None):
   ```

2. **Transformation Workflow**:
   ```python
   def transform(self):
       """Applies ROI transformations to all images in the source Imset"""
       for roi in self.roit.ROI:
           self.roit.roi = roi
           imset_new = self.roit.transform_imset(self.source)
   ```

3. **Path Management**:
   - Automatic target path generation based on transformation parameters
   - Support for custom output paths

### Roit Class

The `Roit` class implements the core transformation logic:

1. **ROI Definitions**:
   ```python
   ROI = ["FFA", "EBA", "VWFA", "OPA", "PPA", "RSC", "V1", "V2", "V3", "V4"]
   ```

2. **Transformation Pipeline**:
   - CLIP feature extraction
   - Modulated embedding generation
   - Stable Diffusion image-to-image transformation
   - IP-Adapter integration

3. **Key Methods**:
   ```python
   def modulated_embedding(self, image_ref):
       """Generates modulated embeddings based on ROI specifications"""
   ```

   ```python
   def transform(self, image_ref):
       """Applies the complete transformation to a single image"""
   ```

   ```python
   def transform_imset(self, imset_ref):
       """Recursively transforms an entire Imset"""
   ```

## Data Structures

### Image Set Management

The system maintains images in a nested dictionary structure that mirrors the original filesystem hierarchy:

```
Imset('root')
  ├── image1.jpg
  ├── image2.png
  └── subfolder
      ├── Imset('subfolder')
      │   ├── image3.jpg
      │   └── image4.png
      └── ...
```

### Region of Interest (ROI) Handling

The system supports the following neuroscience ROIs:

- FFA (Fusiform Face Area)
- EBA (Extrastriate Body Area)
- VWFA (Visual Word Form Area)
- OPA (Occipital Place Area)
- PPA (Parahippocampal Place Area)
- RSC (Retrosplenial Cortex)
- V1-V4 (Visual Cortex areas)

Each ROI can be configured with:
- Maximization/minimization of specific features
- Alpha parameter for interpolation control
- Gamma parameter for transformation strength
- Random seed for reproducibility

## Transformation Pipeline

### Image Processing Workflow

1. **Input Processing**:
   - Image loading and resizing
   - RGB conversion
   - Hierarchical organization

2. **Feature Extraction**:
   - CLIP feature extraction
   - ROI-specific modulation vectors

3. **Transformation**:
   - Spherical linear interpolation (SLERP)
   - Stable Diffusion image-to-image generation
   - IP-Adapter conditioning

4. **Output Generation**:
   - Image saving with preserved structure
   - Automatic naming based on parameters

### Modulated Embedding Generation

The `modulated_embedding` method implements the core neuroscience-inspired transformation:

```python
def modulated_embedding(self, image_ref):
    image_ref_clip = self.clip_extractor(image_ref).detach().cpu().numpy()
    mod_embed = self._load_mod_embed()
    endpoint = mod_embed * np.linalg.norm(image_ref_clip)
    embeds = torch.from_numpy(
        slerp(image_ref_clip, endpoint, 1, t0=self.alpha, t1=self.alpha)
    ).unsqueeze(1).to(self.device)[0]
    return embeds
```

This process:
1. Extracts CLIP features from the input image
2. Loads ROI-specific modulation vectors
3. Computes spherical interpolation between original and modulated features
4. Returns embeddings for the diffusion pipeline

## Command Line Interfaces

### ROI-Dataset CLI

The `roid` command line interface provides batch processing capabilities:

```bash
python -m roid -s SOURCE_PATH [-t TARGET_PATH]
```

Parameters:
- `-s/--source`: Path to source folder containing images
- `-t/--target`: Optional path to target folder (default: auto-generated)

### ROI-Transform CLI

The `roit` command line interface provides single image/folder transformation:

```bash
python -m roit -i INPUT_PATH [-o OUTPUT_PATH]
```

Parameters:
- `-i/--input`: Path to input image or folder
- `-o/--output`: Optional path to save output (default: auto-generated)

## Utility Modules

### CLIP Extractor

The `CLIPExtractor` class provides:

1. **Feature Extraction**:
   ```python
   def forward(self, imgs):
       """Extracts CLIP features from PIL Images"""
   ```

2. **Batch Processing**:
   ```python
   def extract_for_dataset(self, dataset: "NaturalScenesDataset"):
       """Processes entire datasets with batching"""
   ```

3. **Model Integration**:
   - Uses LAION's CLIP-ViT-H-14 model
   - Handles preprocessing and device management

### Spherical Linear Interpolation (SLERP)

The `slerp` function implements vectorized spherical interpolation:

```python
def slerp(v0, v1, num, t0=0, t1=1, DOT_THRESHOLD=0.9995):
    """Vectorized spherical linear interpolation for batches of vectors"""
```

Key features:
- Handles batches of vectors efficiently
- Automatic fallback to linear interpolation for nearly colinear vectors
- Configurable interpolation range

### Execution Time Logging

The `@log_time` decorator provides:

1. **Automatic Timing**:
   ```python
   @log_time
   def some_function():
       # function implementation
   ```

2. **Detailed Output**:
   - Function name
   - Execution time in seconds
   - Arguments passed to the function

## Configuration and Setup

### Environment Variables

The system requires the following environment variables:

- `IPADAPTER`: Path to IP-Adapter model files
- `CHECKPOINTS`: Path to ROI-specific model checkpoints
- `MODEMBED`: Path to modulation embedding files

### Model Loading

The `Roit` class handles loading of multiple models:

1. **Diffusion Pipeline**:
   ```python
   def _load_diffusion_pipeline(self):
       """Loads Stable Diffusion v1.5 with DDIM scheduler"""
   ```

2. **IP-Adapter**:
   ```python
   def _load_ip_adapter(self):
       """Initializes IP-Adapter with CLIP model"""
   ```

3. **CLIP Extractor**:
   ```python
   def _load_clip(self):
       """Initializes CLIP feature extractor"""
   ```

4. **ROI-Specific Models**:
   ```python
   def _load_dino_encoder(self):
       """Loads ROI-specific DINO encoders"""
   ```

## Examples and Usage

### Basic Transformation

```python
from roid.roid import Roid
from roit.roit import Roit

# Initialize transformation engine
roit = Roit(roi="FFA", maximize=True, alpha=0.7, gamma=0.6)

# Create ROI dataset processor
roid = Roid(source_path="path/to/images", roit=roit)

# Apply transformations
roid.transform()

# Save results
roid.target.save()
```

### Command Line Usage

Process an entire directory structure:
```bash
python -m roid -s ./input_images -t ./output_images
```

Transform a single image:
```bash
python -m roit -i input.jpg -o output.jpg
```

## Appendices

### Supported Image Formats

The system supports the following image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

### ROI Definitions

| ROI | Full Name | Function |
|-----|-----------|----------|
| FFA | Fusiform Face Area | Face recognition |
| EBA | Extrastriate Body Area | Body part recognition |
| VWFA | Visual Word Form Area | Word recognition |
| OPA | Occipital Place Area | Scene recognition |
| PPA | Parahippocampal Place Area | Place recognition |
| RSC | Retrosplenial Cortex | Spatial navigation |
| V1-V4 | Visual Cortex Areas | Basic to complex visual processing |

### Configuration File Example

Example `.env` file:
```
IPADAPTER=/path/to/ip_adapter
CHECKPOINTS=/path/to/checkpoints
MODEMBED=/path/to/mod_embeddings
```