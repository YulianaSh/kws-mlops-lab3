"""Перевірка моделі на стабільних еталонних зразках (golden samples).

Кожен зразок із expected.json надсилається на /predict; якщо хоча б один
розпізнано неправильно — скрипт завершується з кодом 1.
"""
import argparse
import json
import sys
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--expected", default="samples/expected.json")
    ap.add_argument("--out", default="metrics/verify.json")
    args = ap.parse_args()

    expected = json.loads(Path(args.expected).read_text())
    base = Path(args.expected).parent
    results, passed = [], 0
    for fname, label in expected.items():
        with open(base / fname, "rb") as f:
            r = requests.post(f"{args.url}/predict", files={"file": (fname, f, "audio/wav")}, timeout=30)
        r.raise_for_status()
        js = r.json()
        ok = js["label"] == label
        passed += ok
        results.append({"file": fname, "expected": label, "predicted": js["label"],
                        "confidence": js["confidence"], "ok": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {fname:10s} expected={label:5s} "
              f"predicted={js['label']:5s} confidence={js['confidence']:.3f}")

    summary = {"passed": passed, "total": len(expected), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"golden samples: {passed}/{len(expected)} passed")
    sys.exit(0 if passed == len(expected) else 1)


if __name__ == "__main__":
    main()
