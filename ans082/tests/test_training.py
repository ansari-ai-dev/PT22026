"""End-to-end checks of the dataset, model and training loop on a tiny synthetic dataset."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASSES, SmartSortDataset, build_transforms  # noqa: E402
from model_loader import build_model  # noqa: E402
from train import train  # noqa: E402

TEST_CLASSES = ["battery", "glass", "paper"]


@pytest.fixture()
def tiny_dataset(tmp_path):
    """24 small coloured images (8 per class): 4 train, 2 val and 2 test per class."""
    root = tmp_path / "images"
    rows = []
    for class_index, name in enumerate(TEST_CLASSES):
        (root / name).mkdir(parents=True)
        for i in range(8):
            rel_path = f"{name}/{name}{i}.jpg"
            colour = (40 * class_index + 10, 90, 160 - 30 * class_index)
            Image.new("RGB", (90, 70), color=colour).save(root / rel_path)
            split = "train" if i < 4 else "val" if i < 6 else "test"
            rows.append({"path": rel_path, "label": name, "split": split, "orig_class": name})
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, root


def _config(tmp_path, csv_path, root, epochs):
    return {
        "experiment": "unit_test",
        "seed": 42,
        "data": {"split_csv": str(csv_path), "data_root": str(root),
                 "image_size": 64, "batch_size": 4, "num_workers": 0},
        "model": {"name": "resnet50", "pretrained": False, "freeze_backbone": False},
        "train": {"epochs": epochs, "lr": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.0,
                  "fp16": False, "class_weights": True, "label_smoothing": 0.0},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt"),
                   "results_dir": str(tmp_path / "out")},
    }


def test_dataset_returns_only_the_requested_split(tiny_dataset):
    csv_path, root = tiny_dataset
    _, eval_tf = build_transforms([0.5] * 3, [0.5] * 3, 64)
    dataset = SmartSortDataset(csv_path, root, "val", eval_tf)
    assert len(dataset) == 6
    image, label = dataset[0]
    assert image.shape == (3, 64, 64)
    assert 0 <= label < len(CLASSES)


def test_model_output_has_one_score_per_class():
    model, mean, std = build_model({"name": "resnet50", "pretrained": False})
    model.eval()
    with torch.no_grad():
        scores = model(torch.zeros(2, 3, 64, 64))
    assert scores.shape == (2, len(CLASSES))
    assert len(mean) == 3 and len(std) == 3


def test_training_writes_log_checkpoints_and_summary(tiny_dataset, tmp_path):
    csv_path, root = tiny_dataset
    summary = train(_config(tmp_path, csv_path, root, epochs=2))
    log = pd.read_csv(tmp_path / "out" / "unit_test_log.csv")
    assert list(log["epoch"]) == [1, 2]
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert (tmp_path / "ckpt" / "last.pt").exists()
    assert (tmp_path / "out" / "unit_test_summary.json").exists()
    assert summary["trainable_parameters"] > 0


def test_resume_continues_after_the_last_finished_epoch(tiny_dataset, tmp_path):
    csv_path, root = tiny_dataset
    train(_config(tmp_path, csv_path, root, epochs=1))
    train(_config(tmp_path, csv_path, root, epochs=2), resume=True)
    log = pd.read_csv(tmp_path / "out" / "unit_test_log.csv")
    assert list(log["epoch"]) == [1, 2]
