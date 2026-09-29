from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_group_metrics(frame, metric, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(frame["subgroup"].astype(str), frame[metric])
    ax.set_ylabel(metric)
    ax.set_xlabel("Subgroup")
    ax.set_title(f"{metric} by subgroup")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reliability(calibration, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = calibration["bins"]
    x = [b["confidence"] for b in bins]
    y = [b["accuracy"] for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_explanation_grid(images, maps, path, titles=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(7, max(3, 3 * n)))
    if n == 1:
        axes = np.array([axes])
    for i, (image, heatmap) in enumerate(zip(images, maps)):
        axes[i, 0].imshow(np.transpose(image, (1, 2, 0)).clip(0, 1))
        axes[i, 1].imshow(np.transpose(image, (1, 2, 0)).clip(0, 1))
        axes[i, 1].imshow(heatmap, alpha=0.45)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        if titles:
            axes[i, 0].set_title(titles[i])
            axes[i, 1].set_title("Attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
