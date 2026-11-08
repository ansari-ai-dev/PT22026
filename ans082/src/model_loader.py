"""Model construction (ResNet-50 baseline now; ViT is added in Phase 6)."""
from torch import nn
from torchvision import models

from data import CLASSES


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """ImageNet-pre-trained ResNet-50 with a new classification layer.

    Returns (model, mean, std). Mean and std come from the pretrained weights' own
    preprocessing preset, not from guessed numbers.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    preset = weights.transforms()
    model = models.resnet50(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    return model, list(preset.mean), list(preset.std)


def build_model(model_cfg: dict, num_classes: int = len(CLASSES)):
    """Build the model named in the config. Returns (model, mean, std)."""
    name = model_cfg["name"]
    if name == "resnet50":
        return build_resnet50(
            num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
