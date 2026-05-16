FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --root-user-action=ignore --upgrade pip \
    && pip install --root-user-action=ignore -r requirements.txt

COPY app.py .
COPY templates ./templates

EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers ${WEB_CONCURRENCY:-2} --threads 4 --timeout 120 app:app
