FROM python:3.11-slim

# Create a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install system dependencies (must be done as root)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user server/ ./server/
COPY --chown=user inference.py .
COPY --chown=user models.py .
COPY --chown=user client.py .
COPY --chown=user openenv.yaml .
COPY --chown=user README.md .

# Expose port for HF Spaces
EXPOSE 7860

# Health check (matching reference project pattern)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command: run the FastAPI app (for HF Spaces)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
