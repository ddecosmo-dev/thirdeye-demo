FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV DATA_DIR=/data \
    MAX_UPLOAD_BYTES=52428800 \
    MAX_UNCOMPRESSED_BYTES=104857600 \
    MAX_FILES_PER_ZIP=500

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
