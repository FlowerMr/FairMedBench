import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def classification_metrics(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    average = "binary" if n_classes == 2 else "macro"
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, average=average, zero_division=0)),
    }
    if n_classes == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        except ValueError:
            result["roc_auc"] = float("nan")
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        result["sensitivity"] = float(tp / (tp + fn)) if tp + fn else float("nan")
        result["specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")
    else:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
        except ValueError:
            result["roc_auc"] = float("nan")
        result["sensitivity"] = float("nan")
        result["specificity"] = float("nan")
    return result
