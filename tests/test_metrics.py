import numpy as np
from fairmedbench.metrics.classification import classification_metrics
from fairmedbench.metrics.calibration import expected_calibration_error, brier_score
from fairmedbench.fairness.metrics import fairness_metrics


def test_classification_metrics():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    result = classification_metrics(y, p)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


def test_calibration_outputs():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    ece, rows = expected_calibration_error(y, p)
    assert 0 <= ece <= 1
    assert isinstance(rows, list)
    assert 0 <= brier_score(y, p) <= 1


def test_fairness_gap():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.4, 0.6]])
    result = fairness_metrics(y, p, np.array(["A", "A", "B", "B"]))
    assert "equal_opportunity_gap" in result
