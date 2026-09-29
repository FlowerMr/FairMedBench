import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, probabilities, n_bins: int = 10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence <= right if right == 1 else confidence < right)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            mean_acc = float(correctness[mask].mean())
            ece += count / len(y_true) * abs(mean_conf - mean_acc)
            rows.append((left, right, count, mean_conf, mean_acc))
    return float(ece), rows


def brier_score(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if probabilities.shape[1] == 2:
        return float(brier_score_loss(y_true, probabilities[:, 1]))
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_curve(y_true, probabilities, n_bins: int = 10):
    ece, rows = expected_calibration_error(y_true, probabilities, n_bins)
    return {
        "ece": ece,
        "bins": [
            {"left": float(r[0]), "right": float(r[1]), "count": r[2], "confidence": r[3], "accuracy": r[4]}
            for r in rows
        ],
    }
