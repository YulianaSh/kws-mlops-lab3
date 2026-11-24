"""Навчання + оцінка KWS-моделі; результат — артефакти в --out.

python -m src.train --classes yes no up down --epochs 20 --seed 42 --out artifacts
"""
import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DEFAULT_URL, SpeechCommands, download_dataset
from src.evaluate import evaluate
from src.model import ARCHITECTURE, KWSNet, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default="data/speech_commands")
    ap.add_argument("--data-url", default=DEFAULT_URL)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    download_dataset(args.data_dir, args.classes, args.data_url)
    train_ds = SpeechCommands(args.data_dir, args.classes, "train", augment=True)
    val_ds = SpeechCommands(args.data_dir, args.classes, "val")
    test_ds = SpeechCommands(args.data_dir, args.classes, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=256)

    model = KWSNet(len(args.classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{ARCHITECTURE}; params={count_params(model)}", flush=True)

    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False, disable=True):
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        val_acc, _ = evaluate(model, val_dl, len(args.classes))
        history.append({"epoch": epoch, "loss": round(loss_sum / total, 4),
                        "train_acc": round(100 * correct / total, 2), "val_acc": round(val_acc, 2)})
        print(f"epoch {epoch:2d}/{args.epochs} loss={loss_sum / total:.4f} "
              f"train_acc={100 * correct / total:.2f}% val_acc={val_acc:.2f}%", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, DataLoader(test_ds, batch_size=256), len(args.classes))

    torch.save({"state_dict": model.state_dict(), "n_mels": 40}, out / "model.pt")
    (out / "labels.json").write_text(json.dumps(args.classes))
    metrics = {
        "model_version": args.version,
        "architecture": ARCHITECTURE,
        "params": count_params(model),
        "model_size_kb": round((out / "model.pt").stat().st_size / 1024, 1),
        "accuracy": round(test_acc, 2),
        "best_val_accuracy": round(best_val, 2),
        "confusion_matrix": cm,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "epochs": args.epochs, "seed": args.seed,
        "torch_version": torch.__version__, "python": platform.python_version(),
        "train_time_s": round(time.time() - t0, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"test accuracy = {test_acc:.2f}%  model.pt = {metrics['model_size_kb']} KB  "
          f"-> {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
