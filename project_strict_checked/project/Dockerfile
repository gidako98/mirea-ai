FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIE_SERVICE_HOST=0.0.0.0 \
    AIE_SERVICE_PORT=8000

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m src.train --config configs/config.json

EXPOSE 8000
CMD ["python", "-m", "src.service", "--config", "configs/config.json"]
