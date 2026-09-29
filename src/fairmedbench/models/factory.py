import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(self.dropout(x))


class TinyViT(nn.Module):
    def __init__(self, num_classes: int = 10, image_size: int = 64, patch_size: int = 8):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        n = (image_size // patch_size) ** 2
        dim = 64
        self.patch = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        attended, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return self.head(self.dropout(tokens[:, 0]))


def build_model(name: str, num_classes: int, pretrained: bool = False, image_size: int = 224):
    name = name.lower()
    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    if name == "tiny_vit":
        return TinyViT(num_classes=num_classes, image_size=image_size)
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, num_classes))
        return model
    if name == "vit_tiny":
        import timm
        return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes, drop_rate=0.1)
    raise ValueError(f"Unsupported model: {name}")
