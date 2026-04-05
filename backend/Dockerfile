FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

COPY scripts/download_models.py /app/scripts/download_models.py

CMD ["sh", "-c", "python scripts/download_models.py & uvicorn app.main:app --host 0.0.0.0 --port 8000"]
