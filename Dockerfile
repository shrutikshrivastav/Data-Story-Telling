FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000

# Shell form so $PORT expands correctly
CMD gunicorn -b 0.0.0.0:$PORT app:app
