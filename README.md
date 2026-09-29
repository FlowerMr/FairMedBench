# FairMedBench

**FairMedBench** is a modular research benchmark for evaluating medical image classification models beyond overall accuracy.

The framework evaluates models from multiple perspectives:

- predictive performance
- subgroup fairness analysis
- explainable AI (XAI)
- uncertainty and calibration
- robustness evaluation
- reproducible experiment configuration
- machine-readable research reports

FairMedBench is designed as a research-oriented Python framework for **Medical AI, Trustworthy AI, Responsible AI, Explainable AI, and Fair Machine Learning** research.

> **Disclaimer:** FairMedBench is a research and benchmarking framework. It is not a medical device, clinical decision-support system, or diagnostic tool.

---

# Key Features

### Model Evaluation
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Sensitivity
- Specificity

### Fairness Analysis
- subgroup performance comparison
- Equal Opportunity Difference
- demographic parity analysis
- transparent fairness reporting

### Explainable AI
Support for:
- Grad-CAM
- Integrated Gradients
- SHAP (where applicable)
- transformer-based explanation analysis

Including:
- explanation visualization
- faithfulness evaluation
- stability analysis

### Uncertainty and Calibration
- Temperature scaling
- Monte Carlo Dropout
- Expected Calibration Error (ECE)
- Brier Score
- reliability analysis

### Robustness
Controlled perturbation evaluation:

- Gaussian noise
- brightness changes
- contrast changes
- small rotations

---

# Research Motivation

Medical AI systems should not be evaluated only by overall accuracy.

A model with high accuracy may still:

- perform differently across demographic groups
- provide unstable explanations
- produce poorly calibrated confidence estimates
- fail under small input changes

FairMedBench provides a unified framework to analyze these properties together.

---

# Main Architecture

```text
                    FairMedBench
                         |
        +----------------+----------------+
        |                |                |
      Data             Models          Evaluation
        |                |                |
 Images + Metadata   CNN / ViT     Performance
                                      Fairness
                                      XAI
                                      Calibration
                                      Robustness
                         |
                    Reporting
                         |
              CSV + JSON + Figures
```

---

# Repository Structure

```text
FairMedBench/
│
├── configs/
│   ├── demo.yaml
│   └── ddi.yaml
│
├── docs/
│
├── scripts/
│   ├── run_demo.py
│   └── validate_ddi.py
│
├── src/fairmedbench/
│   ├── data/
│   ├── models/
│   ├── metrics/
│   ├── fairness/
│   ├── explainability/
│   ├── uncertainty/
│   ├── robustness/
│   ├── reporting/
│   └── visualization/
│
├── tests/
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

# Installation

## Requirements

- Python >= 3.10
- Git
- Windows / Linux / macOS

A GPU is not required for the demo.  
A CUDA-enabled GPU is recommended for real experiments with pretrained vision models.

---

## Clone Repository

```bash
git clone https://github.com/FlowerMr/FairMedBench.git
cd FairMedBench
```

---

## Create Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

# Demo

FairMedBench includes a lightweight demo using synthetic data.

Run:

```bash
fairmedbench demo --config configs/demo.yaml
```

The demo validates the complete evaluation pipeline without requiring medical datasets.

Results:

```text
results/demo/
```

Synthetic results should not be interpreted as medical evidence.

---

# DDI Experiment

The main real-data experiment uses the:

**Diverse Dermatology Images (DDI) Dataset**

DDI enables evaluation of dermatology classification models with subgroup analysis, including Fitzpatrick skin-type groups.

Dataset information:

- https://ddi-dataset.github.io/
- https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images

## Dataset Policy

The DDI images are not included in this repository.

Users must obtain the dataset through the official distribution process.

Expected structure:

```text
data/
└── ddi/
    ├── images/
    └── ddi_metadata.csv
```

Validate dataset:

```bash
python scripts/validate_ddi.py
```

---

# Experimental Results

FairMedBench was evaluated on the DDI dataset using two vision architectures:

- ResNet18
- ViT-Tiny

The evaluation considers predictive performance, fairness, explainability, and uncertainty.

## Model Comparison

| Model | Accuracy | Balanced Accuracy | ROC-AUC | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.674 | 0.598 | 0.630 | 0.411 | 0.441 | 0.755 |
| ViT-Tiny | 0.750 | 0.620 | 0.658 | 0.421 | 0.353 | 0.888 |

---

## Fairness Analysis

| Model | Equal Opportunity Gap | Demographic Parity Gap |
|---|---:|---:|
| ResNet18 | 0.311 | 0.162 |
| ViT-Tiny | 0.402 | 0.173 |

Fairness metrics are reported as descriptive subgroup statistics and should be interpreted together with dataset characteristics and clinical assumptions.

---

## XAI and Uncertainty

| Model | XAI Faithfulness | XAI Stability | Predictive Entropy |
|---|---:|---:|---:|
| ResNet18 | 0.080 | 0.676 | 0.342 |
| ViT-Tiny | 0.018 | 0.523 | 0.426 |

---

# Reproducibility

Experiments are controlled using configuration files:

```text
configs/demo.yaml
configs/ddi.yaml
```

Results depend on:

- dataset version
- preprocessing pipeline
- model checkpoint
- evaluation configuration

---

# Limitations

Important limitations:

- subgroup metadata may contain measurement errors
- dataset composition may introduce sampling bias
- fairness metrics do not automatically define clinical fairness
- XAI explanations are not causal explanations
- uncertainty estimates do not guarantee true uncertainty
- results from one dataset should not be directly generalized to other populations
- clinical deployment requires extensive validation

---

# Research Report

A detailed research-style report is available:

```text
docs/research_report.md
```

It covers:

- motivation
- methodology
- experimental setup
- evaluation framework
- results
- limitations
- future directions

---

# Development

Run tests:

```bash
pytest
```

The modular architecture allows adding:

- new datasets
- new models
- new metrics
- new explanation methods

without redesigning the complete pipeline.

---

# License

Released under the MIT License.

External datasets such as DDI remain subject to their original access and usage agreements.

---

# Citation

If you use FairMedBench in academic work, please cite this repository and the original dataset/model sources.
