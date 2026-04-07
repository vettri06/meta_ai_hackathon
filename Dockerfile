FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/ /app/server/
COPY inference.py /app/inference.py
COPY models.py /app/models.py
COPY client.py /app/client.py
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md

# Expose port for HF Spaces
EXPOSE 7860

# Health check (matching reference project pattern)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Default command: run the FastAPI app (for HF Spaces)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
