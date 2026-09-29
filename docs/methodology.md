# Methodology

FairMedBench evaluates an image classifier along four primary axes: predictive performance, subgroup disparity, explanation behavior, and uncertainty/calibration. Robustness is an optional fifth axis.

## Evaluation unit

Each sample is represented by an image path, a target label, a categorical subgroup value, and a stable sample identifier. The framework does not assume that the subgroup is Fitzpatrick skin type. A caller can supply any categorical demographic, clinical, site, or acquisition attribute.

## Predictive performance

For binary classification the benchmark reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, sensitivity, and specificity. For multiclass classification it uses macro averaging for precision, recall, and F1 and one-vs-rest macro ROC-AUC when defined.

## Subgroup disparity

The benchmark computes the same predictive metrics separately for each subgroup and reports the maximum-minus-minimum gap for selected metrics. This is a descriptive comparison, not a universal fairness score.

## Fairness metrics

For binary tasks the framework reports equal opportunity gap using subgroup true-positive rates and demographic parity gap using subgroup positive prediction rates at a 0.5 threshold. These measures are intentionally presented with assumptions and limitations rather than as universal medical fairness criteria.

## Explainability

Integrated Gradients is model-agnostic. Grad-CAM is available for convolutional layers. The project also contains a lightweight attention visualization path for the included TinyViT demonstration model. Explanation evaluation uses deletion-based faithfulness and perturbation stability where the chosen explainer supports the corresponding model.

## Calibration and uncertainty

Expected Calibration Error and Brier score are computed overall. Reliability diagrams show confidence versus empirical accuracy. Temperature scaling is available as a post-hoc calibration method. Monte Carlo Dropout is available for models that contain dropout layers.

## Robustness

The optional robustness module applies controlled noise, brightness, contrast, and rotation changes and compares prediction accuracy with clean evaluation. It is intended as a stress test, not as a substitute for clinical validation.
