"""Зведений звіт пайплайна у Markdown (Job Summary) + summary.json."""
import argparse
import json
import os
from pathlib import Path


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics")
    ap.add_argument("--model-metrics", default="artifacts/metrics.json")
    ap.add_argument("--out", default="metrics/report.md")
    args = ap.parse_args()

    m = load(args.model_metrics) or {}
    lat = load(Path(args.metrics) / "latency.json") or {}
    ver = load(Path(args.metrics) / "verify.json") or {}
    sha = os.getenv("GITHUB_SHA", "local")[:7]
    event = os.getenv("GITHUB_EVENT_NAME", "local")
    min_acc = float(os.getenv("MIN_ACCURACY", "0"))

    summary = {
        "commit": sha, "event": event, "epochs": m.get("epochs"),
        "accuracy": m.get("accuracy"), "min_accuracy": min_acc,
        "model_size_kb": m.get("model_size_kb"),
        "latency_ms": lat.get("latency_ms"),
        "client_latency_ms": lat.get("client_latency_ms"),
        "golden_samples": {"passed": ver.get("passed"), "total": ver.get("total")},
        "train_time_s": m.get("train_time_s"),
    }
    Path(args.metrics).mkdir(parents=True, exist_ok=True)
    (Path(args.metrics) / "summary.json").write_text(json.dumps(summary, indent=2))

    def ok(cond):
        return "✅" if cond else "❌"

    acc = m.get("accuracy")
    gs_ok = ver.get("passed") is not None and ver.get("passed") == ver.get("total")
    L = lat.get("latency_ms") or {}
    C = lat.get("client_latency_ms") or {}
    lines = [
        "## 🎙️ KWS ML-pipeline report",
        "",
        f"**Commit:** `{sha}` · **Event:** `{event}` · **Epochs:** {m.get('epochs', '—')}",
        "",
        "| Метрика | Значення | Статус |",
        "|---|---|---|",
        f"| Accuracy (test) | {acc if acc is not None else '—'} % (поріг {min_acc:g} %) | "
        f"{ok(acc is not None and acc >= min_acc)} |",
        f"| Golden samples | {ver.get('passed', '—')}/{ver.get('total', '—')} | {ok(gs_ok)} |",
        f"| Latency mean / p50 / p95 | {L.get('mean', '—')} / {L.get('p50', '—')} / {L.get('p95', '—')} мс | |",
        f"| Client latency mean / p95 | {C.get('mean', '—')} / {C.get('p95', '—')} мс | |",
        f"| Розмір моделі | {m.get('model_size_kb', '—')} КБ | |",
        f"| Параметрів | {m.get('params', '—')} | |",
        f"| Час навчання | {m.get('train_time_s', '—')} с | |",
        "",
    ]
    if ver.get("results"):
        lines += ["### Golden samples", "", "| Файл | Очікувано | Прогноз | Впевненість | |", "|---|---|---|---|---|"]
        lines += [f"| {r['file']} | {r['expected']} | {r['predicted']} | {r['confidence']:.3f} | {ok(r['ok'])} |"
                  for r in ver["results"]]
        lines.append("")
    if m.get("confusion_matrix"):
        labels = json.loads(Path(args.model_metrics).with_name("labels.json").read_text())
        lines += ["### Матриця помилок (test)", "", "| істина \\ прогноз | " + " | ".join(labels) + " |",
                  "|---" * (len(labels) + 1) + "|"]
        lines += [f"| **{c}** | " + " | ".join(map(str, row)) + " |" for c, row in zip(labels, m["confusion_matrix"])]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
