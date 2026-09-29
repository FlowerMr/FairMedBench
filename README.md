# FairMedBench

**FairMedBench** is a modular research benchmark for evaluating medical image classification models beyond overall accuracy.

The framework combines:

- predictive performance
- subgroup performance and fairness analysis
- explainable AI (XAI)
- uncertainty and calibration
- lightweight robustness analysis
- reproducible experiment configuration
- machine-readable reports and visualizations

The project is designed as a research-oriented Python package and portfolio project for work in Medical AI, Trustworthy AI, Responsible AI, Explainable AI, and Fair Machine Learning.

> **Important:** FairMedBench is a research and benchmarking framework. It is not a clinical decision-support system and must not be used for clinical diagnosis.

## Research Questions

FairMedBench is designed around five questions:

1. Do medical image classifiers perform differently across population subgroups?
2. How large are subgroup performance disparities?
3. Do explanation methods behave consistently across subgroups?
4. Are confidence and uncertainty calibrated equally well across subgroups?
5. Can models with similar overall performance have different fairness, explainability, calibration, or robustness profiles?

## Main Architecture

```text
                    FairMedBench
                         |
        +----------------+----------------+
        |                |                |
      Data             Models          Evaluation
        |                |                |
   Images +            CNN / ViT      Performance
   Metadata                           Fairness
                                      Calibration
                                      XAI
                                      Robustness
        |                |                |
        +----------------+----------------+
                         |
                    Reporting
                         |
          CSV + JSON + Figures + Reports
```

## Repository Structure

```text
FairMedBench/
├── configs/
│   ├── demo.yaml
│   └── ddi.yaml
├── docs/
│   ├── explainability.md
│   ├── fairness.md
│   ├── limitations.md
│   ├── methodology.md
│   ├── research_report.md
│   └── uncertainty.md
├── notebooks/
├── scripts/
│   ├── run_demo.py
│   └── validate_ddi.py
├── src/fairmedbench/
│   ├── data/
│   ├── explainability/
│   ├── fairness/
│   ├── metrics/
│   ├── models/
│   ├── pipeline/
│   ├── reporting/
│   ├── robustness/
│   ├── uncertainty/
│   ├── visualization/
│   ├── demo.py
│   ├── ddi_experiment.py
│   └── cli.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.10 or newer
- Git
- A Windows, Linux, or macOS environment
- Internet access for installing Python packages

A GPU is **not required for the demo**. A CUDA-capable GPU is recommended for the real DDI experiment, especially when using pretrained vision models and XAI methods.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/FairMedBench.git
cd FairMedBench
```

If you are running the project from the downloaded ZIP instead, extract it and open a terminal in the `FairMedBench` folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

```bash
python -m pip install --upgrade pip
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

For notebooks:

```bash
pip install -e ".[dev,notebook]"
```

## Run the Demo

The repository contains a lightweight demo that does not require the restricted DDI images.

Run:

```bash
fairmedbench demo --config configs/demo.yaml
```

Alternatively:

```bash
python -m fairmedbench.cli demo --config configs/demo.yaml
```

The demo creates a small synthetic dataset and exercises the main evaluation pipeline.

Results are written under:

```text
results/demo/
```

The demo is intended to verify software functionality. Its synthetic subgroup results must not be interpreted as medical evidence.

## Run Tests

```bash
pytest
```

The tests cover configuration loading, dataset handling, metric calculations, and important edge cases.

## DDI Dataset

The primary real-data experiment is designed for the **Diverse Dermatology Images (DDI)** dataset.

DDI is useful for this project because it contains dermatology images together with demographic information that enables subgroup analysis, including Fitzpatrick skin-type groups.

Official dataset information:

- Stanford AIMI DDI dataset
- DDI project website: https://ddi-dataset.github.io/
- Stanford AIMI dataset page: https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images

### Dataset policy

Do **not** commit the DDI images to this repository.

The repository expects the user to obtain the dataset through its official distribution process and place it locally.

Expected layout:

```text
data/
└── ddi/
    ├── images/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── ddi_metadata.csv
```

The exact metadata column names should be checked with the supplied validation script before running the experiment.

Validate the local dataset with:

```bash
python scripts/validate_ddi.py
```

### DDI experiment configuration

The default DDI configuration is stored in:

```text
configs/ddi.yaml
```

It defines the image location, metadata location, image size, batch size, number of epochs, models, and output directory.

Run the experiment with:

```bash
fairmedbench ddi --config configs/ddi.yaml
```

or:

```bash
python -m fairmedbench.cli ddi --config configs/ddi.yaml
```

The experiment is designed to compare two model families using a shared evaluation protocol. The purpose is not to declare one architecture universally superior, but to compare their performance, subgroup behavior, calibration, explainability, and robustness.

## Evaluation

### Predictive performance

The framework supports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable
- Sensitivity
- Specificity

Metrics are reported overall and by subgroup.

### Fairness and subgroup analysis

The framework reports transparent subgroup comparisons rather than collapsing all findings into an arbitrary universal fairness score.

Depending on the task and available labels, the analysis can include:

- performance gaps
- true-positive-rate differences / Equal Opportunity Difference
- demographic-parity analysis when its assumptions are appropriate

Fairness metrics are accompanied by methodological limitations because no single fairness definition is universally appropriate for medical applications.

### Explainable AI

The project provides a modular explanation layer intended to support methods such as:

- Grad-CAM
- Integrated Gradients
- SHAP where practical
- transformer-oriented explanation methods such as attention-based analysis

The framework is designed not only to visualize explanations but also to evaluate them quantitatively where feasible.

The XAI analysis includes:

- explanation visualization
- perturbation/deletion-style faithfulness analysis
- stability under small input perturbations
- subgroup-level comparison of explanation behavior

### Uncertainty and calibration

The uncertainty module includes:

- temperature scaling
- Monte Carlo Dropout
- Expected Calibration Error (ECE)
- Brier Score
- reliability diagrams
- confidence and uncertainty distributions

Calibration can be evaluated overall and separately for subgroups.

### Robustness

A lightweight robustness component can evaluate controlled perturbations such as:

- Gaussian noise
- brightness changes
- contrast changes
- small rotations

The purpose is to examine how model performance changes under controlled perturbations and whether the degradation differs between subgroups.

## Output

Experiments generate machine-readable results and visualizations under the configured output directory.

Typical outputs include:

```text
results/
└── <experiment>/
    ├── overall_metrics.csv
    ├── subgroup_metrics.csv
    ├── fairness_metrics.csv
    ├── calibration_metrics.csv
    ├── xai_metrics.csv
    ├── robustness_metrics.csv
    ├── figures/
    └── report/
```

Exact files depend on the experiment configuration and the availability of the relevant model/XAI components.

## Reproducibility

The project uses configuration files and explicit random seeds so experiments can be reproduced more consistently.

The main configuration files are:

```text
configs/demo.yaml
configs/ddi.yaml
```

Results should always be interpreted together with the dataset version, preprocessing configuration, model checkpoint, and evaluation configuration used to produce them.

## Limitations

This project has several important limitations:

- subgroup metadata can contain measurement errors
- dataset composition can introduce sampling bias
- fairness metrics measure statistical properties and do not automatically establish clinical fairness
- XAI explanations should not be treated as causal explanations
- uncertainty estimates do not guarantee true clinical uncertainty
- results from one dataset should not automatically be generalized to other populations or clinical settings
- the demo uses synthetic data and is not evidence about medical model behavior
- clinical deployment requires substantially more validation than this research benchmark provides

For more detail, see `docs/limitations.md`.

## Research Report

A research-style report is provided in:

```text
docs/research_report.md
```

The report follows a mini-paper structure covering motivation, research questions, methodology, experimental setup, results, limitations, and future work.

## Development

Run tests with:

```bash
pytest
```

The project is organized as a Python package so that new datasets, models, metrics, and explanation methods can be added without rewriting the complete pipeline.

## License

This software is released under the MIT License. See `LICENSE` for details.

The license of the FairMedBench software does not change the terms under which external datasets such as DDI may be accessed or used. Always follow the dataset provider's terms.

## Citation

If this repository is used in academic work, please cite the repository and the specific dataset/model sources used in the experiment.

## Disclaimer

FairMedBench is intended for research and educational purposes. It is not a medical device, diagnostic system, or clinical decision-support tool.
