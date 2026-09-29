import torch
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution


def _target_from_output(logits):
    return int(logits.argmax(dim=1)[0].item())


def integrated_gradients(model, image, target=None, steps=32):
    model.eval()
    image = image.requires_grad_(True)
    with torch.no_grad():
        logits = model(image)
    target = _target_from_output(logits) if target is None else int(target)
    method = IntegratedGradients(model)
    attribution = method.attribute(image, target=target, n_steps=steps)
    return attribution.detach()


def gradcam(model, layer, image, target=None):
    model.eval()
    with torch.no_grad():
        logits = model(image)
    target = _target_from_output(logits) if target is None else int(target)
    method = LayerGradCam(model, layer)
    attribution = method.attribute(image, target=target)
    return LayerAttribution.interpolate(attribution, image.shape[-2:]).detach()


def attention_rollout(model, image, target=None):
    model.eval()
    tokens = model.patch(image).flatten(2).transpose(1, 2)
    cls = model.cls.expand(image.shape[0], -1, -1)
    tokens = torch.cat([cls, tokens], dim=1) + model.pos
    normalized = model.norm1(tokens)
    _, weights = model.attn(
        normalized,
        normalized,
        normalized,
        need_weights=True,
        average_attn_weights=True,
    )
    identity = torch.eye(weights.shape[-1], device=weights.device).unsqueeze(0)
    fused = weights + identity
    fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    cls_attention = fused[:, 0, 1:]
    side = int(cls_attention.shape[1] ** 0.5)
    return cls_attention.reshape(image.shape[0], 1, side, side).detach()


def attribution_to_map(attribution):
    return attribution.abs().sum(dim=1, keepdim=True)
