<a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
</a>

<a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg?style=for-the-badge" alt="CC BY 4.0">
</a>

<a href="https://github.com/ss8319/ForensicSex">
    <img src="https://img.shields.io/badge/GitHub-ss8319%2FForensicSex-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
</a>

Copyright © 2026 <a href="https://orcid.org/0009-0000-1701-7747">Shamus Sim Zi Yang <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0002-9207-0385">Tyrone Chen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>, <a href="https://orcid.org/0000-0002-8432-9035">Choy Ker Woon <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

Code in this repository is provided under a [MIT license](https://opensource.org/licenses/MIT). This documentation is provided under a [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/).

# Forensic Sex Identification

Forensic Sex identification from radiographs with Convolutional Neural Networks.

## Quick Start

### Prerequisites
- Python 3.11 or higher
- `uv` package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ForensicSex
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```
   This creates a virtual environment and installs all dependencies.

3. **Activate the virtual environment**
   
   **On Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Verify installation**
   ```bash
   python --version
   ```

### Running Notebooks

Once the environment is activated, you can run Jupyter notebooks:

```bash
jupyter notebook
```

Or use `uv run` to execute notebooks directly without activating:

```bash
uv run jupyter notebook
```

## Project Structure

### Core Components

| Component | Type | Description |
|-----------|------|-------------|
| `src/train.py` | Script | Unified training script for AP/Lateral views with sweep support |
| `src/train_cross_validation.py` | Script | 9-fold stratified cross-validation for both views |
| `grad-cam/Grad-CAM.ipynb` | Notebook | Model interpretability using Grad-CAM heatmaps |
| `notebooks/visualise_data_augmentation.ipynb` | Notebook | Visualize data transformations and augmentations |
| `hyperparams.json` | Config | Central configuration for all hyperparameters |
| `pyproject.toml` | Config | Project dependencies and metadata (uv/PEP 517) |

### Key Training Features
- **Architecture Support**: ResNet18, ResNet34, DenseNet121
- **Imbalance Handling**: Toggleable between Weighted Loss and Weighted Sampling
- **Optimization**: Early stopping, StepLR, and ReduceLROnPlateau support
- **Tracking**: Integrated WandB sweeps and Azure ML logging

## Training Models

The training scripts support loading configurations from a JSON file (default: `hyperparams.json`) while allowing command-line overrides. This is the most streamlined way to run experiments.

### Standard Training

```bash
# Run using default hyperparams.json
uv run python src/train.py

# Run with overrides
uv run python src/train.py --learning_rate 0.005 --model_type resnet34
```

### Cross-Validation Training

```bash
# Run using default hyperparams.json
uv run python src/train_cross_validation.py

# Run with overrides
uv run python src/train_cross_validation.py --use_weighted_loss
```

### Streamlined Workflow

1.  **Configure**: Edit `hyperparams.json` at the project root to set your baseline parameters.
2.  **Train**: Run `uv run python src/train.py` to train a single model.
3.  **Validate**: Run `uv run python src/train_cross_validation.py` to perform 9-fold cross-validation with the same parameters.

### Training Options (CLI Overrides)

| Parameter | Description | Default in JSON |
|-----------|-------------|-----------------|
| `--config` | Path to custom JSON config | `hyperparams.json` |
| `--view` | Radiograph view (ap/lateral) | `ap` |
| `--model_type` | Architecture (resnet18/resnet34/densenet121) | `resnet18` |
| `--use_weighted_loss` | Use weighted loss (flag) | `false` |
| `--scheduler_type` | Scheduler (step/plateau) | `step` |
| `--learning_rate` | Base learning rate | `0.01` |

*Note: Any command-line argument provided will override the value in the JSON configuration file.*

### Hyperparameter Sweep with WandB

Perform automated hyperparameter optimization:

```bash
# Run WandB sweep with 30 trials using defaults
uv run python src/train.py --use_wandb --wandb_sweep
```

## Training Strategies for Class Imbalance

Both training scripts support **two approaches** to handle class imbalance, easily toggleable via the `--use_weighted_loss` flag:

### Approach 1: Weighted Loss (Default)
Use `--use_weighted_loss` flag to enable class-weighted CrossEntropyLoss:
- **What it does**: Applies higher loss penalties to misclassifications of minority classes
- **How**: Weights are calculated as inverse class frequencies
- **Example**: `python src/train.py --view ap --input_data "raw data/ap" --use_weighted_loss`

### Approach 2: Weighted Sampling
Omit the `--use_weighted_loss` flag to use WeightedRandomSampler:
- **What it does**: Oversamples minority classes during training
- **How**: Samples are drawn with probability proportional to inverse class frequency
- **Example**: `python src/train.py --view ap --input_data "raw data/ap"`
