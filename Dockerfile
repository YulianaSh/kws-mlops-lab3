# ================= Target: trainer =================
# Образ для навчання: модель навчається під час `docker run`, а не `docker build`,
# і зберігається у змонтований каталог /artifacts.
FROM python:3.12-slim AS trainer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--classes", "yes", "no", "up", "down", "--epochs", "10", \
     "--seed", "42", "--out", "/artifacts"]

# ================= Target: runtime =================
# Образ сервісу: лише залежності інференсу, код і модель з каталогу artifacts/.
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
# Модель береться з каталогу artifacts/ (артефакт job train), а не з етапу trainer
COPY artifacts/ ./artifacts/

ARG GIT_SHA=unknown
ARG MODEL_VERSION=1.0.0
ENV MODEL_PATH=/app/artifacts/model.pt \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
