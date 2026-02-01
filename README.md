# PDE-Constrained Optimization for Neural Image Segmentation with Physics Priors

This repository contains the implementation for the research paper **[PDE-Constrained Optimization for Neural Image Segmentation with Physics Priors](https://drive.google.com/file/d/1D-BM5eC0vfxKh8Fnj6tGj98nn-1osmQx/view?usp=sharing)**. The project introduces a novel approach to cell segmentation by incorporating partial differential equation (PDE) constraints as physics-based regularization in neural network training.

## Overview

This project implements a PDE-constrained neural network for cell segmentation that combines:
- **Reaction-Diffusion (RD) PDE**: Models boundary evolution and sharpening
- **Phase-Field Energy**: Enforces smooth interfaces and boundary regularization
- **Two-Stage Training**: Baseline training followed by PDE-constrained fine-tuning

The method demonstrates improved segmentation performance, especially in low-data regimes, by leveraging physics priors encoded through PDE constraints.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/seemapoudel58/Physics_informed_image_segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Data Setup

The dataset for this project is available at:
**[Google Drive - Physics Informed Image Segmentation](https://drive.google.com/drive/folders/1sJjH_z2qHfOjzOxbtjjC3T5iJwK1vusE?usp=sharing)**

1. Download the `images.zip` file from the Google Drive link
2. Extract the archive to the project root directory
3. The expected directory structure should be:
```
cell_segmentation/
├── images/
│   ├── training/
│   ├── validation/
│   ├── in_dist_testing/
│   ├── out_dist_testing/
│   └── annotation/
│       ├── training_annotation.json
│       ├── validation_annotation.json
│       ├── in_dist_testing_annotation.json
│       └── out_dist_testing_annotation.json
```

## Usage

### Training a Model

#### Two-Stage Training (Recommended)

The default training strategy uses two stages:
1. **Stage I**: Baseline training (Dice + BCE loss)
2. **Stage II**: PDE-constrained fine-tuning (Dice + BCE + PDE regularization)

```bash
python main.py
```

#### Custom Training Options

```bash
# Single-stage training (PDE from start)
python main.py --single-stage

# Low-data training (10% of training data)
python main.py --train-fraction 0.1

# Custom PDE weights
python main.py --pde-weight 1e-4 --phase-field-weight 1e-4

# Custom PDE parameters
python main.py --diffusion-coeff 5.0 --reaction-threshold 0.5 --epsilon 0.05

# Custom training hyperparameters
python main.py --batch-size 8 --learning-rate 1e-4 --stage1-epochs 50 --stage2-epochs 50
```

#### Training Arguments

- `--single-stage`: Use single-stage training (PDE from start) instead of two-stage
- `--pde-weight`: Weight for Reaction-Diffusion PDE regularization λ_RD (default: 1e-4)
- `--phase-field-weight`: Weight for phase-field energy λ_PF (default: 1e-4)
- `--diffusion-coeff`: Diffusion coefficient D (default: 5.0)
- `--reaction-threshold`: Reaction term threshold a (default: 0.5)
- `--epsilon`: Interface width parameter ε (default: 0.05)
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Learning rate for AdamW optimizer (default: 1e-4)
- `--stage1-epochs`: Maximum epochs for Stage I (default: 50)
- `--stage2-epochs`: Maximum epochs for Stage II (default: 50)
- `--early-stopping-patience`: Patience for early stopping (default: 5)
- `--train-fraction`: Fraction of training data to use (e.g., 0.1 for 10%)
- `--seed`: Random seed for reproducibility (default: 42)

### Running Ablation Studies

The project includes comprehensive ablation studies to evaluate the contribution of different components:

#### Available Ablation Studies

**Results Studies:**
- **R1**: Influence of PDE Constraints (100% Data) - Component ablation comparing Baseline, RD Only, Phase-Field Only, and RD+Phase-Field
- **R2**: Low Sample Regime Analysis - Full model with varying data fractions (10%, 25%, 50%, 75%, 100%)
- **R3**: Influence of PDE Constraints (10% Data) - Component ablation with 10% training data

**Sensitivity Analysis:**
- **S1**: Reaction Threshold Sensitivity Analysis - Tests reaction threshold values (0.3, 0.4, 0.5, 0.6, 0.7)
- **S2**: Diffusion Coefficient Sensitivity Analysis - Tests diffusion coefficients (0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
- **S3**: Interface Width Sensitivity Analysis - Tests epsilon values (0.001, 0.01, 0.05, 0.1, 0.2)

#### Running a Single Ablation Study

```bash
# Run a specific ablation study
python run_ablation.py --ablation R1

# Run all ablation studies
python run_ablation.py --ablation all

# Custom paths and hyperparameters
python run_ablation.py --ablation S1 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --stage1-epochs 50 \
    --stage2-epochs 50
```

#### Ablation Study Arguments

- `--ablation`: Ablation study to run (R1, R2, R3, S1, S2, S3, or 'all')
- `--train-dir`: Training images directory (default: 'images/training')
- `--train-json`: Training annotations JSON (default: 'images/annotation/training_annotation.json')
- `--val-dir`: Validation images directory (default: 'images/validation')
- `--val-json`: Validation annotations JSON (default: 'images/annotation/validation_annotation.json')
- `--in-dist-test-dir`: In-distribution test images directory
- `--in-dist-test-json`: In-distribution test annotations JSON
- `--out-dist-test-dir`: Out-of-distribution test images directory
- `--out-dist-test-json`: Out-of-distribution test annotations JSON
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--stage1-epochs`: Max epochs for stage 1 (default: 50)
- `--stage2-epochs`: Max epochs for stage 2 (default: 50)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--output-dir`: Output directory for results (default: 'output/ablation')

### Model Evaluation

Evaluate trained models on test sets:

```bash
python evaluate.py --model-path <path_to_model.pth> \
    --test-dir images/in_dist_testing \
    --test-json images/annotation/in_dist_testing_annotation.json
```



## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{poudel_khadka_2026,
  title={PDE-Constrained Optimization for Neural Image Segmentation with Physics Priors},
  author={Poudel, Seema K. and Khadka, Sunny K.},
  booktitle={Proceedings of the 1st International Conference on Statistics, Data Science and Optimization (ICSDO-2026)},
  pages={},
  year={2026},
  note={Presented at ICSDO-2026, January 30--31, 2026}
}
```



