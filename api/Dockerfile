# File: api/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY api/requirements_api.txt /app/api/requirements_api.txt
RUN pip install --no-cache-dir -r /app/api/requirements_api.txt

COPY . /app
COPY outputs/saved_models/roberta_multiclass /app/outputs/saved_models/roberta_multiclass

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
