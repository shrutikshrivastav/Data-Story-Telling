# Use the specified Python slim image for a lightweight footprint
FROM python:3.11.9-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if required (e.g., for pandas/numpy optimization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (Render sets PORT dynamically, but 5000 is a good fallback)
EXPOSE 5000

# Start the application using Gunicorn for production readiness
CMD ["sh", "-c", "gunicorn --workers=2 --threads=4 --worker-class=gthread --bind 0.0.0.0:${PORT:-5000} app:app"]
