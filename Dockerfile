# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent bytecode generation and ensure unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create non-root system group and user for least-privilege execution
RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser

# Set working directory
WORKDIR /app

# Install dependencies first to optimize Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only minimal files required for inference
COPY api.py ./
COPY inference_helper.py ./
COPY validation.py ./
COPY risk_scoring.py ./
COPY explainability.py ./
COPY models/hybrid_logistic_regression.joblib ./models/hybrid_logistic_regression.joblib

# Set ownership of application files
RUN chown -R appuser:appgroup /app

# Run as non-root user
USER appuser

# Expose standard application port
EXPOSE 8000

# Container healthcheck using standard library urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Launch FastAPI inference service via uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
