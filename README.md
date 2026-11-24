# KWS ML-pipeline (ЛР №3)

CI/CD/CT-пайплайн у GitHub Actions для аудіокласифікатора голосових команд
**yes / no / up / down** (Google Speech Commands, CNN з ЛР №1–2).

```
pull_request / push / workflow_dispatch
        │
        ▼
      lint ──► train ──► verify ──► build-push
                 │          │            │
                 ▼          ▼            ▼
          артефакт      артефакт     образ у GHCR
            model        report    (push у main або
                                    ручний запуск)
```

| Тригер | Епох | Поріг accuracy | Публікація образу |
|---|---|---|---|
| `pull_request` у main | 2 | 60 % | ні |
| `push` у main | 10 | 80 % | так (`sha-<commit>`, `latest`) |
| `workflow_dispatch` | з форми (за замовч. 10) | 80 % | якщо `push_image = true` |

## Локальний запуск тих самих кроків
```bash
docker build --target trainer -t kws-trainer .
docker run --rm -v "$PWD/artifacts:/artifacts" kws-trainer --epochs 10 --seed 42 --out /artifacts
docker build --target runtime -t kws-api .
docker run -d --name kws -p 8000:8000 kws-api
python scripts/verify_model.py --url http://localhost:8000 --expected samples/expected.json
python scripts/benchmark.py --url http://localhost:8000 --n 100 --out metrics/latency.json
python scripts/make_report.py --metrics metrics --out metrics/report.md
```

## Образ
```bash
docker pull ghcr.io/yulianash/kws-mlops-lab3/kws-api:latest
docker run -d -p 8000:8000 ghcr.io/yulianash/kws-mlops-lab3/kws-api:latest
curl -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
```
