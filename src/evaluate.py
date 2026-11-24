"""Оцінка accuracy збереженої моделі на тестовому наборі.

python -m src.evaluate --artifacts artifacts --data-dir data/speech_commands
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import SpeechCommands, download_dataset
from src.model import KWSNet


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    cm = [[0] * n_classes for _ in range(n_classes)]
    for x, y in loader:
        for t, p in zip(y.tolist(), model(x).argmax(1).tolist()):
            cm[t][p] += 1
    total = sum(map(sum, cm))
    acc = 100 * sum(cm[i][i] for i in range(n_classes)) / max(total, 1)
    return acc, cm


def load_model(artifacts: Path):
    ckpt = torch.load(artifacts / "model.pt", map_location="cpu")
    labels = json.loads((artifacts / "labels.json").read_text())
    model = KWSNet(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data-dir", default="data/speech_commands")
    args = ap.parse_args()

    model, labels = load_model(Path(args.artifacts))
    download_dataset(args.data_dir, labels)
    ds = SpeechCommands(args.data_dir, labels, "test")
    acc, cm = evaluate(model, DataLoader(ds, batch_size=256), len(labels))
    print(f"Тестових прикладів: {len(ds)}")
    print(f"Accuracy (test): {acc:.2f}%")
    print("Матриця помилок (рядки - істина, стовпці - прогноз):")
    print("        " + "".join(f"{c:>7}" for c in labels))
    for c, row in zip(labels, cm):
        print(f"{c:>7} " + "".join(f"{v:>7}" for v in row))


if __name__ == "__main__":
    main()
