import numpy as np
import pandas as pd

from fairmedbench.metrics.classification import classification_metrics


def subgroup_report(y_true, probabilities, subgroups):
    frame = pd.DataFrame({"y_true": y_true, "subgroup": subgroups})
    probabilities = np.asarray(probabilities)
    frame["row"] = np.arange(len(frame))
    rows = []
    for group, part in frame.groupby("subgroup", dropna=False):
        idx = part["row"].to_numpy()
        metrics = classification_metrics(part["y_true"].to_numpy(), probabilities[idx])
        rows.append({"subgroup": str(group), "n": int(len(part)), **metrics})
    return pd.DataFrame(rows)


def fairness_metrics(y_true, probabilities, subgroups):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    subgroups = np.asarray(subgroups)
    if probabilities.shape[1] != 2:
        return {"note": "Equal opportunity and demographic parity are implemented for binary tasks only."}
    pred = probabilities[:, 1] >= 0.5
    rows = []
    for group in np.unique(subgroups):
        mask = subgroups == group
        positives = pred[mask]
        actual = y_true[mask]
        tpr = float(((pred[mask]) & (actual == 1)).sum() / max((actual == 1).sum(), 1))
        positive_rate = float(positives.mean()) if len(positives) else float("nan")
        rows.append((str(group), tpr, positive_rate))
    if not rows:
        return {}
    tprs = [r[1] for r in rows]
    rates = [r[2] for r in rows]
    return {
        "equal_opportunity_gap": float(max(tprs) - min(tprs)),
        "demographic_parity_gap": float(max(rates) - min(rates)),
        "group_true_positive_rates": {r[0]: r[1] for r in rows},
        "group_positive_prediction_rates": {r[0]: r[2] for r in rows},
        "interpretation": "Gaps are descriptive statistics and should be interpreted with task-specific clinical and statistical assumptions."
    }
