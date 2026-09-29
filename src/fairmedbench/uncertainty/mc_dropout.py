import torch


def _enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, loader, passes: int = 10, device: str = "cpu"):
    model = model.to(device)
    was_training = model.training
    model.eval()
    _enable_dropout(model)
    outputs = []
    labels = []
    groups = []
    ids = []
    for _ in range(passes):
        probabilities = []
        current_labels = []
        current_groups = []
        current_ids = []
        for images, y, subgroup, sample_ids in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            current_labels.extend(y.tolist())
            current_groups.extend(list(subgroup))
            current_ids.extend(list(sample_ids))
        outputs.append(torch.cat(probabilities))
        if not labels:
            labels = current_labels
            groups = current_groups
            ids = current_ids
    if was_training:
        model.train()
    mean_probability = torch.stack(outputs).mean(0)
    predictive_entropy = -(mean_probability * mean_probability.clamp_min(1e-8).log()).sum(1)
    return {
        "probabilities": mean_probability,
        "predictive_entropy": predictive_entropy,
        "labels": labels,
        "subgroups": groups,
        "sample_ids": ids,
    }
