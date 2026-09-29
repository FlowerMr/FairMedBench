import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _confidence(model, image):
    return torch.softmax(model(image), dim=1).max(dim=1).values


def deletion_faithfulness(model, image, attribution, fractions=(0.1, 0.25, 0.5)):
    model.eval()
    attribution = attribution.abs().sum(dim=1, keepdim=True)
    flat = attribution.flatten(1)
    base = _confidence(model, image)
    values = []
    for fraction in fractions:
        k = max(1, int(flat.shape[1] * fraction))
        threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(-1, 1, 1, 1)
        mask = (attribution >= threshold).float()
        perturbed = image * (1.0 - mask)
        confidence = _confidence(model, perturbed)
        values.append(float((base - confidence).mean().item()))
    return float(np.mean(values))


def explanation_stability(explainer, image, noise_std=0.01):
    image2 = (image + torch.randn_like(image) * noise_std).clamp(0, 1)
    a = explainer(image)
    b = explainer(image2)
    a = a.flatten(1)
    b = b.flatten(1)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return float((a * b).sum(dim=1).mean().item())
