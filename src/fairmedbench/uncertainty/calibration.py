import torch
from torch import nn


def fit_temperature(logits, labels, max_iter: int = 50):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp_min(0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(0.05).item())


def temperature_scale(logits, temperature: float):
    return torch.softmax(torch.as_tensor(logits) / max(float(temperature), 0.05), dim=1)
