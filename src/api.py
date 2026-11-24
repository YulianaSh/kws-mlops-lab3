"""FastAPI-сервіс інференсу KWS-моделі.

Конфігурація через змінні середовища: MODEL_PATH, MODEL_VERSION, LOG_LEVEL.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import wav_to_features
from src.model import ARCHITECTURE, KWSNet, count_params

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
ARTIFACTS = MODEL_PATH.parent
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kws-api")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель завантажується ОДИН раз під час старту сервісу
    t0 = time.perf_counter()
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
    labels = json.loads((ARTIFACTS / "labels.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    model = KWSNet(len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")["state_dict"])
    model.eval()
    with torch.no_grad():  # прогрів
        model(torch.zeros(1, 1, 40, 101))
    state.update(model=model, labels=labels, metrics=metrics,
                 version=os.getenv("MODEL_VERSION", metrics.get("model_version", "1.0.0")))
    log.info("model %s loaded in %.1f ms (%s)", MODEL_PATH, (time.perf_counter() - t0) * 1000, labels)
    yield
    state.clear()


app = FastAPI(title="KWS API", version="1.0.0", lifespan=lifespan,
              description="Розпізнавання голосових команд yes / no / up / down (ЛР №3)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    m = state["metrics"]
    return {
        "model_version": state["version"],
        "architecture": ARCHITECTURE,
        "classes": state["labels"],
        "params": count_params(state["model"]),
        "model_size_kb": round(MODEL_PATH.stat().st_size / 1024, 1),
        "accuracy": m["accuracy"],
        "epochs": m["epochs"],
        "seed": m["seed"],
        "trained_at": m["trained_at"],
        "torch_version": torch.__version__,
        "git_sha": os.getenv("GIT_SHA", "unknown"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Порожній файл")
    t0 = time.perf_counter()
    try:
        x = wav_to_features(data).unsqueeze(0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Некоректний WAV-файл: {type(exc).__name__}")
    with torch.no_grad():
        probs = torch.softmax(state["model"](x), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000  # лише препроцесинг + інференс
    idx = int(probs.argmax())
    labels = state["labels"]
    return {
        "label": labels[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(labels, probs)},
        "latency_ms": round(latency_ms, 2),
        "model_version": state["version"],
    }
