<h1 align="center">Count Anything</h1>

<p align="center">
  <strong>Text-guided object counting across domains with instance-grounded point predictions</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/MengqiLei/count-anything"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/spaces/MengqiLei/count-anything-demo"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-orange" alt="Hugging Face Demo"></a>
  <a href="https://arxiv.org/abs/2605.30846"><img src="https://img.shields.io/badge/arXiv-2605.30846-b31b1b" alt="arXiv Paper"></a>
  <a href="assets/paper/Count-Anything.pdf"><img src="https://img.shields.io/badge/Paper-PDF-blue" alt="Paper PDF"></a>
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <img src="assets/readme/count_anything_poster.jpg" alt="Count Anything poster" width="100%">
</p>

<p align="center">
  <strong>Play with the demo: <a href="https://huggingface.co/spaces/MengqiLei/count-anything-demo">🔗Huggingface Demo</a></strong>
</p>

## Overview

This repository introduces **Count Anything**, a generalist model for text-guided object counting across domains. Given an image and a natural-language query, Count Anything returns an instance-grounded set of target points whose cardinality gives the count. This formulation unifies category-conditioned counting with interpretable spatial localization.

### Cross-domain Text-guided Counting

- Study text-guided object counting across domains, where users specify the target with a category name or a natural-language query.
- Construct **CLOC**, a Cross-domain Large-scale Object Counting dataset that reorganizes diverse public data sources into a unified counting benchmark.
- Cover six visual domains: General Scene, Remote Sensing, Histopathology, Cellular Microscopy, Agriculture, and Microbiology.

### Dual-granularity Instance Enumeration

- Adopt discrete instance points as the final prediction form, rather than using density maps as the final output.
- Use a **Region-level Sparse Counter (RSC)** to provide object-level anchoring for large and sparse targets.
- Use a **Pixel-level Dense Counter (PDC)** to capture small, crowded, and weakly bounded targets through dense point prediction.

### Point-centric Supervision and Complementary Fusion

- Convert heterogeneous annotations, including boxes, points, polygons, masks, rotated boxes, and label maps, into counting points with optional boxes.
- Use point-centric supervision so every valid instance is supervised by a point, while bounding boxes are used only when reliable box annotations exist.
- Combine RSC and PDC with **Complementary Count Fusion (CCF)** in a parameter-free manner, suppressing duplicate counts while preserving their complementarity.

Count Anything is trained and evaluated on CLOC, which contains about 220K images, 619 categories, and 15M object instances. Extensive experiments show that Count Anything achieves strong counting accuracy and multi-domain generalization, substantially outperforming existing open-world counting methods.

## Main Results

![Main CLOC comparison table from the paper](assets/readme/main_results_cloc_table.png)

## Quick Start

### 1. Environment Setup

Create a conda environment and install the required dependencies:

```bash
conda create -n countanything python=3.12 -y
conda activate countanything
pip install -r requirements.txt
```

The dependency list is intentionally kept minimal. If the default pip resolver does not select the CUDA build you need, please install the PyTorch and torchvision builds that match your local CUDA version.

### 2. Weight Preparation

For inference, validation, or test-only reproduction, only the released CountAnything checkpoint is required. Download `count_anything.pt` from [Hugging Face](https://huggingface.co/MengqiLei/count-anything).

After downloading, place the file at:

```text
checkpoints/count_anything.pt
```

The standalone validation and test configurations load `checkpoints/count_anything.pt` directly.

If you want to train or fine-tune CountAnything from the SAM3 initialization, please also download the official SAM3 pretrained weights. Due to license and redistribution restrictions, this repository does not directly provide the official SAM3 pretrained weights. Please visit the official SAM3 Hugging Face page:

[Hugging Face](https://huggingface.co/facebook/sam3)

Download the SAM3 pretrained weight file `sam3.pt`, and place it at:

```text
pretrained/sam3.pt
```

Please make sure to download the **SAM3** weights, not SAM3.1 weights. The expected file name is:

```text
sam3.pt
```

By default, the training configuration initializes the model from `pretrained/sam3.pt`.

### 3. Data Preparation

This repository uses the CLOC dataset by default. The dataset preparation guide explains how to download the CLOC annotation archive, the distributable augmented-image archive, and the raw images of each source dataset. Please prepare the dataset following that guide before running training or evaluation.

💡 **Dataset availability note.** Because some source datasets used by CLOC are subject to license and redistribution restrictions, we cannot directly release the complete CLOC image set and can only reference those data sources. We release the CLOC annotation files produced by our re-annotation process and the subset of augmented images that can be redistributed. To reproduce the complete CLOC dataset, please download the original source images and follow the preprocessing, rebuilding, and audit steps in [data/README.md](data/README.md). **We can also provide a processed, ready-to-use copy of the complete CLOC dataset upon request; please contact us by email for access.**

The default configurations expect the train, validation, and test annotations at:

```text
data/annotations/train_split_expanded_by_class.json
data/annotations/val_split_expanded_by_class.json
data/annotations/test_split_expanded_by_class.json
```

Each sample corresponds to one image-category counting task. The annotation files provide the image path, category text, and the point / bbox annotations for the corresponding category. The actual image location is specified by the `image_path` field in each annotation record.

The dataset directory is organized as follows:

```text
data/
  annotations/        # CLOC train/val/test JSON files
  images/             # Raw dataset download and extraction directories
  augmented/          # Augmented images referenced by CLOC annotations
  tools/              # Data conversion, augmented-image rebuilding, and audit scripts
  README.md           # English dataset preparation guide
```

For the full dataset construction workflow, including raw dataset downloads, format conversion, augmented-image rebuilding, and path auditing, please refer to [data/README.md](data/README.md).

If your dataset paths differ from the default layout, please edit:

```text
config/count_anything_train_cloc.yaml
config/count_anything_val_cloc.yaml
config/count_anything_test_cloc.yaml
```

and update `paths.train_annotation_file` and `paths.val_annotation_file` so they point to your local annotation files.

### 4. Training

Start training with `train.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash train.sh
```

By default, `train.sh` uses:

```text
config/count_anything_train_cloc.yaml
```

This configuration corresponds to the main model setting in the paper: it initializes from SAM3 pretrained weights, enables RSC, PDC, and CCF, and trains the LoRA parameters together with the counting branch. The LoRA learning rate is `1e-3`; the learning rate follows a 30-epoch cosine schedule with `min_lr_ratio=0.1`.

Validation after each epoch uses:

```text
data/annotations/val_split_expanded_by_class.json
```

The default training parameters include:

- `train_batch_size=18`
- `val_batch_size=40`
- `max_epochs=30`
- `val_epoch_freq=1`
- `visualize_val_every_n_epochs=5`

After training starts, logs, visualizations, and checkpoints are saved by default to:

```text
exp/count_anything_train_cloc/
```

To change data paths, batch size, number of epochs, or the output directory, edit:

```text
config/count_anything_train_cloc.yaml
```

### 5. Validation

Run standalone validation with `val.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash val.sh
```

By default, `val.sh` uses:

```text
config/count_anything_val_cloc.yaml
```

This configuration loads:

```text
checkpoints/count_anything.pt
```

and evaluates on the CLOC validation split:

```text
data/annotations/val_split_expanded_by_class.json
```

Validation logs and prediction statistics are saved by default to:

```text
exp/count_anything_val_cloc/
```

To validate another checkpoint or another validation set, edit:

```text
config/count_anything_val_cloc.yaml
```

and update the checkpoint path and `paths.val_annotation_file`.

### 6. Testing

Evaluate a checkpoint with `test.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash test.sh
```

By default, `test.sh` uses:

```text
config/count_anything_test_cloc.yaml
```

This configuration loads:

```text
checkpoints/count_anything.pt
```

and evaluates on the CLOC test split:

```text
data/annotations/test_split_expanded_by_class.json
```

Evaluation logs and prediction statistics are saved by default to:

```text
exp/count_anything_test_cloc/
```

To evaluate another checkpoint or another test set, edit:

```text
config/count_anything_test_cloc.yaml
```

and update the checkpoint path and `paths.val_annotation_file`.

## Repository Structure

```text
CountAnything/
  train.sh                         # Default training entry point
  val.sh                           # Default validation entry point
  test.sh                          # Default test entry point
  requirements.txt                 # Python dependencies
  config/
    count_anything_train_cloc.yaml # CLOC training configuration
    count_anything_val_cloc.yaml   # CLOC validation configuration
    count_anything_test_cloc.yaml  # CLOC test configuration
  count_anything/
    model/                         # CountAnything model components
    train/                         # Trainer, losses, and matcher
    eval/                          # Post-processing and counting evaluation
  sam3/                            # SAM3-based image-language backbone components
  pretrained/
    sam3.pt                        # Place the SAM3 pretrained weights here
  checkpoints/
    count_anything.pt               # Place the CountAnything checkpoint here
  data/                            # CLOC dataset annotations and preparation tools
  exp/                             # Training and evaluation outputs
```

## Questions and Support

If you encounter any difficulty with dataset preparation, model weights, training, validation, or evaluation, please feel free to contact us and we will do our best to help.

## Citation

If you find Count Anything useful in your research, please consider citing:

```bibtex
@article{lei2026count_anything,
  title={Count Anything},
  author={Lei, Mengqi and Cheng, Shuokun and Bao, Wei and Du, Shaoyi and Yong, Jun-Hai and Li, Siqi and Gao, Yue},
  journal={arXiv preprint arXiv:2605.30846},
  year={2026}
}
```
