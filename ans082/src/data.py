"""Image transforms and loading helpers for SmartSort AI.

Phase 4 provides the transforms. The Dataset and DataLoader code is added in Phase 5.
"""
from PIL import Image
from torchvision import transforms as T

IMAGE_SIZE = 224


def get_normalization(model_name: str = "google/vit-base-patch16-224"):
    """Return (mean, std) from the model's own image processor, not guessed numbers."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)
    return list(processor.image_mean), list(processor.image_std)


def build_transforms(mean, std, image_size: int = IMAGE_SIZE):
    """Return (train_transform, eval_transform) for the given normalisation values."""
    train_tf = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    eval_tf = T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def load_rgb(path) -> Image.Image:
    """Open an image and convert it to RGB (some dataset images are palette mode)."""
    with Image.open(path) as img:
        return img.convert("RGB")