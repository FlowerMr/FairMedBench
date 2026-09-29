import torch


def _transform(images, kind):
    if kind == "noise":
        return (images + 0.05 * torch.randn_like(images)).clamp(0, 1)
    if kind == "brightness":
        return (images * 1.2).clamp(0, 1)
    if kind == "contrast":
        mean = images.mean(dim=(2, 3), keepdim=True)
        return ((images - mean) * 1.2 + mean).clamp(0, 1)
    if kind == "rotation":
        return torch.rot90(images, 1, dims=(-2, -1))
    raise ValueError(f"Unknown perturbation: {kind}")


@torch.no_grad()
def evaluate_robustness(model, loader, device="cpu", perturbations=("noise", "brightness", "contrast")):
    model.eval()
    result = {}
    for kind in perturbations:
        correct = 0
        total = 0
        for images, labels, _, _ in loader:
            images = _transform(images, kind).to(device)
            predictions = model(images).argmax(1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
        result[kind] = correct / total if total else float("nan")
    return result
