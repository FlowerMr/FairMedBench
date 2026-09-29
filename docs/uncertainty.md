# Uncertainty and calibration

Confidence is not the same thing as calibrated probability. FairMedBench therefore separates calibration from predictive performance.

## ECE

Expected Calibration Error partitions predictions by confidence and computes a weighted absolute difference between mean confidence and empirical accuracy.

## Brier score

For binary classification, the Brier score is the mean squared error between predicted probability for the positive class and the binary outcome. The implementation uses the multiclass generalization when needed.

## Temperature scaling

Temperature scaling learns one positive scalar on a validation set and divides logits by that value before softmax. It is a post-hoc calibration method and should not be fitted on the final test set.

## Monte Carlo Dropout

The framework can keep dropout modules active at inference and average multiple stochastic predictions. Predictive entropy is reported as a simple uncertainty summary.
