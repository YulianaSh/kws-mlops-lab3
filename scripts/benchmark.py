"""Вимірювання latency сервісу: N запитів до /predict після розігріву -> JSON.

python scripts/benchmark.py --url http://localhost:8000 --n 100 --warmup 10 --out metrics/latency.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def pct(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summary(values):
    return {"mean": round(statistics.mean(values), 2), "p50": round(pct(values, 50), 2),
            "p95": round(pct(values, 95), 2), "max": round(max(values), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", "-n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--out", default="metrics/latency.json")
    args = ap.parse_args()

    payloads = [(f.name, f.read_bytes()) for f in sorted(Path(args.samples).glob("*.wav"))]
    s = requests.Session()
    for i in range(args.warmup):
        name, data = payloads[i % len(payloads)]
        s.post(f"{args.url}/predict", files={"file": (name, data, "audio/wav")}).raise_for_status()

    server, client = [], []
    for i in range(args.n):
        name, data = payloads[i % len(payloads)]
        t0 = time.perf_counter()
        r = s.post(f"{args.url}/predict", files={"file": (name, data, "audio/wav")})
        client.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        server.append(r.json()["latency_ms"])

    res = {"requests": args.n, "warmup": args.warmup,
           "latency_ms": summary(server),          # препроцесинг + інференс (поле latency_ms)
           "client_latency_ms": summary(client)}   # повний HTTP-запит
    print(json.dumps(res, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
