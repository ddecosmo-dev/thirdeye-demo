FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV DATA_DIR=/data \
    CLOUD_INGEST_URL=http://cloud-service:8080/ingest \
    CAPTURE_FPS_LIMIT=5

EXPOSE 8081

CMD ["python", "-m", "app.main"]
