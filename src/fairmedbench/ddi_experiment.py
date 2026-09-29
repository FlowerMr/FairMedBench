from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from sklearn.model_selection import train_test_split

from fairmedbench.data.ddi import build_ddi_records
from fairmedbench.data.base import TabularImageDataset
from fairmedbench.models.factory import build_model
from fairmedbench.pipeline.evaluate import evaluate_model
from fairmedbench.reporting.report import save_report
from fairmedbench.seed import set_seed
from fairmedbench.visualization.plots import plot_group_metrics, plot_reliability
from fairmedbench.robustness.perturbations import evaluate_robustness
from fairmedbench.uncertainty.calibration import fit_temperature, temperature_scale
from fairmedbench.uncertainty.mc_dropout import mc_dropout_predict


def _train(model, loader, epochs, device):
    import torch.nn as nn
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


def _loader(records, transform, batch_size, shuffle=False):
    return DataLoader(TabularImageDataset(records, transform), batch_size=batch_size, shuffle=shuffle)


def _subgroup_calibration(labels, probabilities, groups):
    from fairmedbench.metrics.calibration import reliability_curve, brier_score
    rows = []
    for group in sorted(set(groups)):
        mask = np.asarray(groups) == group
        cal = reliability_curve(np.asarray(labels)[mask], np.asarray(probabilities)[mask])
        rows.append({
            "subgroup": str(group),
            "n": int(mask.sum()),
            "ece": float(cal["ece"]),
            "brier_score": float(brier_score(np.asarray(labels)[mask], np.asarray(probabilities)[mask])),
        })
    return rows


def _run_xai(model, model_name, loader, device, max_samples=6):
    from fairmedbench.explainability.attribution import integrated_gradients, gradcam
    from fairmedbench.explainability.evaluation import deletion_faithfulness, explanation_stability
    model.eval()
    rows = []
    layer = None
    if model_name == "resnet18":
        layer = model.layer4[-1]
    seen = 0
    for images, labels, groups, ids in loader:
        images = images.to(device)
        for i in range(len(images)):
            if seen >= max_samples:
                return rows
            image = images[i:i + 1].detach().clone().requires_grad_(True)
            target = int(labels[i].item())
            if model_name == "resnet18":
                attribution = gradcam(model, layer, image, target=target)
                stability_explainer = lambda x: gradcam(model, layer, x, target=target).abs().sum(1)
                method = "gradcam"
            else:
                attribution = integrated_gradients(model, image, target=target, steps=16)
                stability_explainer = lambda x: integrated_gradients(model, x, target=target, steps=8).abs().sum(1)
                method = "integrated_gradients"
            faithfulness = deletion_faithfulness(model, image, attribution)
            stability = explanation_stability(stability_explainer, image)
            rows.append({
                "sample_id": str(ids[i]),
                "subgroup": str(groups[i]),
                "method": method,
                "faithfulness": faithfulness,
                "stability": stability,
            })
            seen += 1
    return rows


def _uncertainty_by_group(model, loader, device, passes=8):
    result = mc_dropout_predict(model, loader, passes=passes, device=device)
    entropy = result["predictive_entropy"].numpy()
    groups = np.asarray(result["subgroups"])
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        rows.append({
            "subgroup": str(group),
            "n": int(mask.sum()),
            "mean_predictive_entropy": float(entropy[mask].mean()),
        })
    return rows


def run_ddi(config):
    set_seed(int(config.get("seed", 42)))
    image_root = config["image_root"]
    metadata_path = config["metadata_path"]
    records = build_ddi_records(metadata_path, image_root)
    labels = [r.label for r in records]
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("seed", 42)),
        stratify=labels,
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=float(config.get("validation_fraction", 0.1875)),
        random_state=int(config.get("seed", 42)),
        stratify=train_labels,
    )
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    size = int(config.get("image_size", 224))
    train_transform = Compose([
        Resize((size, size)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    batch_size = int(config.get("batch_size", 16))
    train_loader = _loader(train_records, train_transform, batch_size, shuffle=True)
    val_loader = _loader(val_records, eval_transform, batch_size)
    test_loader = _loader(test_records, eval_transform, batch_size)
    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"
    output = Path(config.get("output_dir", "results/ddi"))
    output.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for model_name in config.get("models", ["resnet18", "vit_tiny"]):
        model = build_model(
            model_name,
            num_classes=2,
            pretrained=bool(config.get("pretrained", True)),
            image_size=size,
        )
        model = _train(model, train_loader, int(config.get("epochs", 5)), device)
        collected, overall, subgroup, fairness, calibration = evaluate_model(model, test_loader, device)

        val_collected = evaluate_model(model, val_loader, device)[0]
        temperature = fit_temperature(val_collected["logits"], val_collected["labels"])
        calibrated_probabilities = temperature_scale(collected["logits"], temperature).numpy()
        calibrated_reliability = calibration.copy()
        from fairmedbench.metrics.calibration import reliability_curve, brier_score
        calibrated_reliability = reliability_curve(collected["labels"], calibrated_probabilities)
        calibrated_reliability["brier_score"] = brier_score(collected["labels"], calibrated_probabilities)
        calibrated_reliability["temperature"] = temperature

        xai_rows = _run_xai(
            model,
            model_name,
            test_loader,
            device,
            max_samples=int(config.get("xai_samples", 6)),
        )
        robustness = evaluate_robustness(
            model,
            test_loader,
            device=device,
            perturbations=tuple(config.get("robustness", ["noise", "brightness", "contrast"])),
        )
        uncertainty = _uncertainty_by_group(
            model,
            test_loader,
            device=device,
            passes=int(config.get("mc_dropout_passes", 8)),
        )

        save_report(
            output / model_name,
            overall,
            subgroup,
            fairness,
            calibrated_reliability,
            robustness=robustness,
            xai=xai_rows,
            uncertainty=uncertainty,
            notes=[
                "DDI images must be obtained directly from the official Stanford AIMI portal under its research-use terms.",
                "Temperature scaling is fitted on a validation split and evaluated on the held-out test split.",
                "XAI metrics are computed on a small configurable sample for practical runtime.",
                "This experiment is not a clinical validation study.",
            ],
        )
        plot_group_metrics(subgroup, "f1", output / model_name / "figures" / "subgroup_f1.png")
        plot_reliability(calibrated_reliability, output / model_name / "figures" / "reliability.png")
        all_summaries[model_name] = {
            **overall,
            "temperature": temperature,
            "mean_xai_faithfulness": float(np.mean([r["faithfulness"] for r in xai_rows])) if xai_rows else float("nan"),
            "mean_xai_stability": float(np.mean([r["stability"] for r in xai_rows])) if xai_rows else float("nan"),
            "mean_predictive_entropy": float(np.mean([r["mean_predictive_entropy"] for r in uncertainty])) if uncertainty else float("nan"),
        }
    pd.DataFrame(all_summaries).T.to_csv(output / "model_comparison.csv")
