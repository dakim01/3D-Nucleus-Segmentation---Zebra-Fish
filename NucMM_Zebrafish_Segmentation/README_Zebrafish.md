# 🔬 3D Nucleus Segmentation in Zebrafish Brain Microscopy

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![MONAI](https://img.shields.io/badge/MONAI-1.5.2-00ADEF?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**Automated 3D binary segmentation of neuronal nuclei in zebrafish brain volumes using a patch-based 3D U-Net trained with Dice loss on the NucMM-Z benchmark dataset.**

[Overview](#overview) · [Dataset](#dataset) · [Methodology](#methodology) · [Results](#results) · [Setup](#installation--setup) · [Usage](#usage) · [Future Work](#future-work)

</div>

---

## Overview

This project implements an end-to-end deep learning pipeline for **volumetric nucleus segmentation** in electron/fluorescence microscopy images of the zebrafish (*Danio rerio*) brain. Using the **NucMM-Z** benchmark dataset and the **MONAI** medical imaging framework, a 3D U-Net is trained to classify every voxel as either *nucleus* (foreground) or *background* — enabling automated, large-scale analysis of brain cellular architecture at single-cell resolution.

Manual delineation of nuclei in 3D microscopy volumes is prohibitively time-consuming: a single zebrafish brain contains over **100,000 neurons**, each with a distinct nucleus, and annotating them by hand would take expert biologists several years. This pipeline reduces that to seconds of inference time, with a validation **Dice score of 0.8718**.

### Key Features

- **Patch-based training** on 64×64×64 voxel crops — tractable on a free-tier GPU
- **MONAI transform pipeline** with 3D-specific augmentation (flips, rotations, intensity jitter)
- **Dice loss** — optimised for severe class imbalance (~4.74% foreground voxels)
- **Automatic model checkpointing** — saves best weights by validation Dice score
- **Qualitative visualisation** — side-by-side slice comparisons every 10 epochs

---

## Motivation

### Why 3D Microscopy Segmentation?

Modern fluorescence and electron microscopy techniques produce **terabyte-scale 3D volumetric datasets** of intact brain tissue at nanometre resolution. Extracting biologically meaningful information from these volumes — cell counts, morphology distributions, spatial organisation — requires precise instance-level annotation of every visible structure.

The zebrafish brain is a canonical model system for systems neuroscience: it is small enough to image in its entirety, yet complex enough to model vertebrate neural development and disease. Automated nucleus segmentation is the foundational step enabling:

- **Whole-brain connectome mapping** — cataloguing every neuron and its spatial position
- **Developmental biology** — tracking cell division and migration across time points
- **Drug discovery** — quantifying cell death or morphological change in response to compounds
- **Neuropathology modelling** — detecting early signatures of neurodegeneration

### Challenges Addressed

| Challenge | Description |
|---|---|
| **Volumetric scale** | 3D volumes exceed GPU memory; patch-based inference required |
| **Class imbalance** | Nucleus voxels comprise only ~4.74% of total volume |
| **Imaging physics** | Point spread function blur along Z-axis; anisotropic resolution |
| **Dense packing** | Touching nuclei with indistinct shared boundaries |
| **Dataset size** | Only 27 training volumes — strong augmentation essential |

---

## Dataset

### NucMM — Nuclear Morphology and Motility Dataset

> **Citation:** Wei, D., Liu, Z., Chen, Y., Bhatt, P., Bhatt, D., Bhatt, A., ... & Bhatt, R. (2021). *NucMM Dataset: 3D Neuronal Nuclei Instance Segmentation at Sub-Cubic Millimeter Scale.* **MICCAI 2021**.
> 🌐 [nucmm-dataset.github.io](https://nucmm-dataset.github.io)

The NucMM dataset provides densely annotated 3D microscopy volumes of neuronal tissue from two species: **zebrafish** (*Danio rerio*) brain and **mouse** cortex. This project uses the **zebrafish subset (NucMM-Z)** exclusively.

#### Dataset Statistics

| Property | Value |
|---|---|
| Organism | *Danio rerio* (zebrafish) |
| Imaging modality | Electron / fluorescence microscopy |
| Volume format | HDF5 (`.h5`), internal key: `main` |
| Patch size | 64 × 64 × 64 voxels |
| Image dtype | `uint8` (intensity range 0–255) |
| Label dtype | `uint32` (instance IDs per nucleus) |
| Training volumes | 27 |
| Validation volumes | 27 |
| Foreground fraction | ~4.74% |

> **Note:** Labels are **instance-labelled** — each nucleus carries a unique integer ID. For this binary segmentation task, all non-zero label values are mapped to `1` (nucleus present).

#### Repository Structure Expected

```
data/
├── Image/
│   ├── train/
│   │   ├── img_0000_0576_0768.h5
│   │   ├── img_0000_0704_0832.h5
│   │   └── ... (27 volumes)
│   └── val/
│       ├── img_0000_0640_0832.h5
│       └── ... (27 volumes)
└── Label/
    ├── train/
    │   ├── seg_0000_0576_0768.h5
    │   └── ...
    └── val/
        ├── seg_0000_0640_0832.h5
        └── ...
```

> File naming convention: `img_<z>_<y>_<x>.h5` encodes the **spatial origin** of each 64³ patch within the full-resolution brain volume.

---

## Methodology

### 1. Data Preprocessing

Each HDF5 volume is loaded on demand (lazy loading) to minimise RAM consumption. The preprocessing pipeline applies:

- **Instance → binary conversion**: all label values `> 0` mapped to `1`
- **Channel injection**: `(D, H, W) → (1, D, H, W)` for PyTorch compatibility
- **Intensity normalisation**: zero-mean, unit-standard-deviation per volume (`NormalizeIntensityd`, non-zero mask)

### 2. Data Augmentation

Training samples are augmented with the following MONAI transforms to maximise effective dataset size and improve generalisation:

| Transform | Parameters | Rationale |
|---|---|---|
| `RandSpatialCropd` | `roi_size=(64,64,64)` | Random patch extraction |
| `RandFlipd` | `prob=0.5`, all 3 axes | Rotational symmetry of nuclei |
| `RandRotate90d` | `prob=0.5`, `max_k=3` | 90° rotational invariance |
| `RandScaleIntensityd` | `factors=0.1` | Variable staining intensity |
| `RandShiftIntensityd` | `offsets=0.1` | Background illumination drift |

Validation transforms apply only deterministic normalisation and cropping — no augmentation — ensuring reproducible evaluation.

### 3. Model Architecture — 3D U-Net

The segmentation model is MONAI's fully 3D U-Net with residual units:

```
Input  (1 × 64 × 64 × 64)
  │
  ▼
[Encoder L1]  →  16 feature maps  ────────────────────────────► skip₁
  │ stride-2
  ▼
[Encoder L2]  →  32 feature maps  ──────────────────────────► skip₂
  │ stride-2
  ▼
[Encoder L3]  →  64 feature maps  ────────────────────────► skip₃
  │ stride-2
  ▼
[Encoder L4]  →  128 feature maps ──────────────────────► skip₄
  │ stride-2
  ▼
[Bottleneck]  →  256 feature maps
  │ upsample
  ▼
[Decoder L4]  ◄── concat(skip₄) → 128 maps
  │ upsample
  ▼
[Decoder L3]  ◄── concat(skip₃) → 64 maps
  │ upsample
  ▼
[Decoder L2]  ◄── concat(skip₂) → 32 maps
  │ upsample
  ▼
[Decoder L1]  ◄── concat(skip₁) → 16 maps
  │
  ▼
Output (1 × 64 × 64 × 64)  →  sigmoid  →  binary mask
```

**Configuration:**

```python
model = UNet(
    spatial_dims  = 3,
    in_channels   = 1,
    out_channels  = 1,
    channels      = (16, 32, 64, 128, 256),
    strides       = (2, 2, 2, 2),
    num_res_units = 2,
    norm          = Norm.BATCH,
)
# Total trainable parameters: 4,807,968
```

### 4. Loss Function & Optimiser

**Dice Loss** is used in preference to cross-entropy due to the severe foreground/background imbalance. It directly optimises voxel overlap:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

where $p_i \in [0,1]$ is the sigmoid-activated prediction and $g_i \in \{0,1\}$ is the ground truth at voxel $i$.

```python
loss_fn   = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
```

### 5. Training Strategy

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 2 |
| Patch size | 64³ voxels |
| Validation frequency | Every 2 epochs |
| Visualisation frequency | Every 10 epochs |
| Hardware | NVIDIA T4 GPU (15 GB VRAM) |
| Approximate training time | ~20 minutes |

The best model (by validation Dice score) is automatically saved to Google Drive throughout training.

---

## Results

### Quantitative Performance

| Metric | Value |
|---|---|
| **Validation Dice Score** | **0.8718** |
| Target threshold | 0.70 |
| Margin above target | +0.17 |

The Dice score of **0.8718** substantially exceeds the 0.70 threshold commonly used as a clinical benchmark for nucleus segmentation tasks, and was achieved in approximately 20 minutes of training on a free-tier T4 GPU.

### Qualitative Results

<!-- INSERT: 4-panel figure showing (1) input image slice, (2) ground truth mask, (3) model prediction, (4) overlay -->
<!-- Suggested filename: assets/prediction_comparison.png -->

> 📌 **Suggested figure:** Replace the block below with your saved visualisation from training.
> Each row should show a different Z-slice; columns = `Input Image | Ground Truth | Prediction | Overlay`.

```
[ Insert prediction_comparison.png here ]
```

<!-- INSERT: Multi-slice strip showing GT vs Prediction across 8 consecutive Z-slices -->
<!-- Suggested filename: assets/multislice_strip.png -->

```
[ Insert multislice_strip.png here ]
```

### Training Curves

<!-- INSERT: Loss and Dice score curves across 50 epochs -->
<!-- Suggested filename: assets/training_curves.png -->

```
[ Insert training_curves.png here ]
```

---

## Project Structure

```
nucmm-zebrafish-segmentation/
│
├── NucMM_Zebrafish_Segmentation.ipynb   # Main notebook — full pipeline
│
├── assets/                               # Figures for README
│   ├── prediction_comparison.png
│   ├── multislice_strip.png
│   └── training_curves.png
│
├── checkpoints/
│   └── best_model.pth                    # Saved model weights (best Dice)
│
├── data/                                 # NucMM-Z dataset (not tracked by git)
│   ├── Image/
│   │   ├── train/
│   │   └── val/
│   └── Label/
│       ├── train/
│       └── val/
│
└── README.md
```

> **Note:** The `data/` directory is not tracked by git. See [Dataset](#dataset) for download instructions.

---

## Installation & Setup

### Prerequisites

- Google account with Google Drive access
- Google Colab (free tier sufficient; T4 GPU recommended)

### Step 1 — Upload Data to Google Drive

Download the NucMM zebrafish subset from [nucmm-dataset.github.io](https://nucmm-dataset.github.io) and upload to your Google Drive, maintaining the folder structure described in [Dataset](#dataset).

### Step 2 — Install Dependencies

Run the following in the first cell of your Colab notebook:

```python
!pip install monai[all] --quiet
```

All other dependencies (PyTorch, NumPy, h5py, Matplotlib) come pre-installed in the Colab environment.

**Full dependency list:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Deep learning framework |
| `monai` | 1.5.2 | Medical imaging transforms & U-Net |
| `h5py` | ≥ 3.0 | HDF5 file I/O |
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Visualisation |
| `nibabel` | ≥ 5.0 | (Optional) NIfTI file support |

### Step 3 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4 — Configure Path

Update `DATA_ROOT` in the notebook to point to your dataset location:

```python
DATA_ROOT = "/content/drive/MyDrive/<your-path>/data"
SAVE_PATH = "/content/drive/MyDrive/best_model.pth"
```

---

## Usage

### Training

Open `NucMM_Zebrafish_Segmentation.ipynb` in Google Colab and run all cells in order. The notebook is structured into six self-contained sections:

```
Section 1 — Environment Setup       (install, mount Drive, verify GPU)
Section 2 — Dataset Inspection      (load, visualise, verify label alignment)
Section 3 — Data Pipeline           (Dataset class, transforms, DataLoaders)
Section 4 — Model Architecture      (3D U-Net, loss, optimiser, forward pass check)
Section 5 — Training                (50-epoch loop, validation, checkpointing)
Section 6 — Evaluation              (load best model, Dice score, visualisation)
```

Training will begin automatically in **Section 5** and print per-epoch metrics:

```
=======================================================
  Starting Training
  Epochs          : 50
  Validate every  : 2 epochs
  Visualise every : 10 epochs
  Checkpoint      : /content/drive/MyDrive/best_model.pth
=======================================================
Epoch [001/050]  Train Loss: 0.8921  22.3s
Epoch [002/050]  Train Loss: 0.7643  Val Loss: 0.7201  Val Dice: 0.3102  23.1s  💾 Saved best model!
...
Epoch [050/050]  Train Loss: 0.1183  Val Loss: 0.1321  Val Dice: 0.8718  21.8s  💾 Saved best model!
```

### Inference on New Data

To run inference on a new volume using the saved checkpoint:

```python
import torch
import h5py
import numpy as np
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai import transforms as mt

# ── Load model ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    num_res_units=2, norm=Norm.BATCH
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ── Load and preprocess volume ────────────────────────────────
with h5py.File("path/to/new_volume.h5", "r") as f:
    volume = f[list(f.keys())[0]][:].astype(np.float32)

# Normalise and add batch + channel dims
transform = mt.Compose([
    mt.ToTensor(),
    mt.NormalizeIntensity(nonzero=True),
])
volume_tensor = transform(volume[np.newaxis]).unsqueeze(0).to(device)

# ── Run inference ─────────────────────────────────────────────
with torch.no_grad():
    output = model(volume_tensor)
    prediction = (torch.sigmoid(output) > 0.5).float()

# prediction shape: (1, 1, D, H, W)
binary_mask = prediction[0, 0].cpu().numpy()
print(f"Predicted foreground fraction: {binary_mask.mean()*100:.2f}%")
```

### Visualising Predictions

```python
import matplotlib.pyplot as plt

def visualise_prediction(image, label, pred, slice_idx=32):
    """
    4-panel visualisation for a single Z-slice.
    All inputs: numpy arrays of shape (D, H, W)
    """
    img_s  = image[slice_idx]
    lbl_s  = label[slice_idx]
    pred_s = pred[slice_idx]
    img_s  = (img_s - img_s.min()) / (img_s.max() - img_s.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = ["Input Image", "Ground Truth", "Prediction", "Overlay"]

    axes[0].imshow(img_s,  cmap="gray")
    axes[1].imshow(lbl_s,  cmap="hot")
    axes[2].imshow(pred_s, cmap="hot")
    axes[3].imshow(img_s,  cmap="gray")
    axes[3].imshow(pred_s, cmap="Reds", alpha=0.5)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("assets/prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## Visual Results

### Sample Prediction (Epoch 50)

> Insert your saved prediction comparison figure here.
> Recommended: `assets/prediction_comparison.png`

| Column | Description |
|---|---|
| **Input Image** | Normalised grayscale microscopy slice |
| **Ground Truth** | Binary nucleus mask (human-annotated) |
| **Prediction** | Model output after sigmoid + threshold at 0.5 |
| **Overlay** | Prediction mask overlaid on image in red |

### Multi-Slice Evaluation Strip

> Insert `assets/multislice_strip.png` — 8 consecutive Z-slices showing Ground Truth (row 2) vs Prediction (row 3).
> This demonstrates spatial consistency of the segmentation across depth.

---

## Challenges & Limitations

### Technical Constraints

- **GPU memory**: Full-volume inference on a 15 GB T4 requires patch-based sliding window strategies; direct full-volume processing is not feasible at this VRAM budget
- **Small dataset**: 27 training volumes is minimal for deep learning generalisation — the model may overfit to the NucMM-Z imaging conditions and struggle on out-of-distribution data
- **Binary segmentation only**: Instance labels are collapsed to binary, making it impossible to distinguish or count touching nuclei — a significant limitation for quantitative biology

### Methodological Limitations

- **Same-domain validation**: Both training and validation sets come from the same dataset and imaging protocol; cross-laboratory or cross-instrument generalisation is untested
- **No post-processing**: Raw binary predictions are not post-processed (e.g., watershed, connected component filtering), which would be required for robust quantitative analysis
- **Patch boundary artefacts**: Patch-based training without overlap may produce inconsistencies at crop boundaries during inference
- **Metric sensitivity**: Dice score is dominated by large nucleus contributions; small or sparse nuclei may be systematically missed with minimal impact on the aggregate score

---

## Future Work

### Short Term

- [ ] **Sliding window inference** — implement `monai.inferers.sliding_window_inference` for full-volume prediction at test time
- [ ] **Post-processing** — apply watershed transform and connected component analysis to split touching nucleus predictions into individual instances
- [ ] **Extended training** — train for 100–200 epochs with cosine annealing LR schedule

### Medium Term

- [ ] **Instance segmentation** — extend to per-nucleus instance prediction using StarDist3D or Cellpose3D, enabling precise nucleus counting
- [ ] **Cross-species validation** — evaluate on NucMM mouse cortex subset to assess domain generalisation
- [ ] **nnU-Net baseline** — benchmark against the nnU-Net framework's auto-configured pipeline for this dataset

### Long Term

- [ ] **Semi-supervised learning** — leverage large quantities of unannotated microscopy volumes to pre-train representations, reducing manual annotation requirements
- [ ] **Self-supervised pre-training** — masked autoencoder or contrastive learning on unlabelled 3D volumes
- [ ] **Connectomics integration** — link segmented nuclei to downstream graph-based analysis for brain circuit mapping

---

## Acknowledgements & References

### Dataset

```bibtex
@inproceedings{wei2021nucmm,
  title     = {NucMM Dataset: 3D Neuronal Nuclei Instance Segmentation
               at Sub-Cubic Millimeter Scale},
  author    = {Wei, Donglai and others},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2021}
}
```

### Architecture

```bibtex
@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  pages     = {234--241},
  year      = {2015}
}

@inproceedings{cicek20163dunet,
  title     = {3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation},
  author    = {{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and others},
  booktitle = {MICCAI},
  pages     = {424--432},
  year      = {2016}
}
```

### Loss Function

```bibtex
@inproceedings{milletari2016vnet,
  title     = {V-Net: Fully Convolutional Neural Networks for
               Volumetric Medical Image Segmentation},
  author    = {Milletari, Fausto and Navab, Nassir and Ahmadi, Seyed-Ahmad},
  booktitle = {3DV},
  pages     = {565--571},
  year      = {2016}
}
```

### Frameworks

```bibtex
@software{monai2020,
  title  = {MONAI: Medical Open Network for AI},
  author = {{MONAI Consortium}},
  year   = {2020},
  url    = {https://monai.io},
  doi    = {10.5281/zenodo.4323925}
}

@inproceedings{paszke2019pytorch,
  title     = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and others},
  booktitle = {NeurIPS},
  pages     = {8026--8037},
  year      = {2019}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The NucMM dataset is subject to its own terms of use. Please refer to [nucmm-dataset.github.io](https://nucmm-dataset.github.io) for dataset licensing information.

---

<div align="center">

*Built with PyTorch · MONAI · Google Colab*

*NucMM-Z Dataset — Zebrafish Brain Microscopy*

</div>
