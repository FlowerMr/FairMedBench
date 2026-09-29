from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from fairmedbench.data.base import ImageRecord, TabularImageDataset
from fairmedbench.models.factory import build_model
from fairmedbench.pipeline.evaluate import evaluate_model
from fairmedbench.reporting.report import save_report
from fairmedbench.seed import set_seed
from fairmedbench.visualization.plots import plot_group_metrics, plot_reliability


def _prepare_demo(root, seed=42):
    root = Path(root)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.images, digits.target, test_size=0.25, random_state=seed, stratify=digits.target
    )
    records = []
    for split, images, labels in [("train", x_train, y_train), ("test", x_test, y_test)]:
        for i, (image, label) in enumerate(zip(images, labels)):
            group = "bright" if image.mean() >= 4 else "dark"
            rgb = np.repeat((image / 16.0)[..., None], 3, axis=2)
            pil = Image.fromarray((rgb * 255).astype(np.uint8)).resize((64, 64))
            path = image_root / f"{split}_{i}.png"
            pil.save(path)
            records.append((split, ImageRecord(str(path), int(label), group, f"{split}_{i}")))
    return records


def _train(model, loader, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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


def run_demo(config):
    set_seed(int(config.get("seed", 42)))
    root = Path(config.get("demo_data_dir", "data/demo"))
    records = _prepare_demo(root, int(config.get("seed", 42)))
    transform = Compose([ToTensor()])
    train_records = [r for split, r in records if split == "train"]
    test_records = [r for split, r in records if split == "test"]
    train_loader = DataLoader(TabularImageDataset(train_records, transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(TabularImageDataset(test_records, transform), batch_size=128)
    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"
    output = Path(config.get("output_dir", "results/demo"))
    output.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for model_name in ["tiny_cnn", "tiny_vit"]:
        model = build_model(model_name, num_classes=10, image_size=64)
        model = _train(model, train_loader, int(config.get("epochs", 2)), device)
        collected, overall, subgroup, fairness, calibration = evaluate_model(model, test_loader, device)
        save_report(output / model_name, overall, subgroup, fairness, calibration, notes=[
            "Demo uses sklearn digits converted to RGB and a synthetic acquisition subgroup based on mean image intensity.",
            "Demo outputs are engineering-pipeline validation results, not medical evidence."
        ])
        plot_group_metrics(subgroup, "f1", output / model_name / "figures" / "subgroup_f1.png")
        plot_reliability(calibration, output / model_name / "figures" / "reliability.png")
        summaries[model_name] = overall
    pd.DataFrame(summaries).T.to_csv(output / "model_comparison.csv")
