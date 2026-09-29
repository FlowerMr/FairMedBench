import numpy as np
import torch
from fairmedbench.metrics.classification import classification_metrics
from fairmedbench.metrics.calibration import brier_score, reliability_curve
from fairmedbench.fairness.metrics import subgroup_report, fairness_metrics


@torch.no_grad()
def collect_predictions(model, loader, device="cpu"):
    model.eval()
    probs = []
    labels = []
    groups = []
    ids = []
    logits = []
    for images, y, subgroup, sample_ids in loader:
        out = model(images.to(device))
        logits.append(out.cpu())
        probs.append(torch.softmax(out, dim=1).cpu())
        labels.extend(y.tolist())
        groups.extend(list(subgroup))
        ids.extend(list(sample_ids))
    return {
        "labels": np.asarray(labels),
        "probabilities": torch.cat(probs).numpy(),
        "logits": torch.cat(logits).numpy(),
        "subgroups": np.asarray(groups),
        "sample_ids": np.asarray(ids),
    }


def evaluate_model(model, loader, device="cpu"):
    collected = collect_predictions(model, loader, device)
    overall = classification_metrics(collected["labels"], collected["probabilities"])
    subgroup = subgroup_report(collected["labels"], collected["probabilities"], collected["subgroups"])
    fairness = fairness_metrics(collected["labels"], collected["probabilities"], collected["subgroups"])
    calibration = reliability_curve(collected["labels"], collected["probabilities"])
    calibration["brier_score"] = brier_score(collected["labels"], collected["probabilities"])
    return collected, overall, subgroup, fairness, calibration
