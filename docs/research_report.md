# FairMedBench Research Report

## Abstract

FairMedBench is a modular evaluation framework for medical image classification. It evaluates predictive performance together with subgroup disparities, explanation behavior, calibration, uncertainty, and optional robustness. The framework is designed to separate reusable evaluation infrastructure from any one dataset.

## Introduction

Medical image models can achieve strong aggregate performance while exhibiting heterogeneous behavior across clinically relevant subgroups. A useful benchmark therefore needs to expose subgroup-level metrics and confidence behavior rather than relying on a single aggregate score.

## Research Questions

1. Do predictive metrics differ across subgroups?
2. How large are the observed subgroup gaps?
3. Does explanation behavior vary across subgroups?
4. Is confidence calibrated similarly across subgroups?
5. Can models with similar aggregate performance behave differently under these additional criteria?

## Methodology

The software accepts image paths, labels, subgroup metadata, and a PyTorch model. It computes overall and subgroup metrics, transparent fairness statistics, calibration measures, and optional explanation and robustness analyses.

## Experimental Setup

The intended real-data experiment uses the Diverse Dermatology Images dataset. The default repository execution uses a lightweight synthetic engineering demo so that the repository can be tested without redistributing medical images.

## Results

Real medical results are intentionally not included unless the experiment has been executed on the user's local DDI copy. The demo produces machine-readable outputs under `results/demo`.

## Discussion

The benchmark is designed to make evaluation multidimensional. Any scientific conclusion must be based on actual runs, sample sizes, uncertainty estimates, and the intended clinical context.

## Limitations

See `docs/limitations.md`.

## Future Work

Add bootstrap confidence intervals, patient-level repeated splits, external validation, subgroup calibration curves, statistical tests with pre-specified hypotheses, and additional explanation methods.

## Conclusion

FairMedBench provides a reusable foundation for trustworthy medical AI evaluation without reducing the research question to aggregate classification accuracy.
