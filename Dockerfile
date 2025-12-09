# OriginHub Agentic API - Dockerfile
# Optimized for OpenAI backend deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/agentic/requirements.txt /app/requirements-agentic.txt

# Create optimized requirements for OpenAI backend (no CUDA/llama-cpp needed)
# Filter out llama-cpp-python and keep everything else
RUN cat /app/requirements-agentic.txt | \
    grep -v "llama-cpp-python" | \
    sed 's/\[cuda\]//' > /app/requirements.txt && \
    echo "" >> /app/requirements.txt && \
    echo "# OpenAI API client" >> /app/requirements.txt && \
    echo "openai>=1.0.0" >> /app/requirements.txt && \
    echo "" >> /app/requirements.txt && \
    echo "# HTTP client for health checks" >> /app/requirements.txt && \
    echo "requests>=2.31.0" >> /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src/ /app/src/
COPY pytest.ini /app/pytest.ini

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "src/agentic/scripts/api_server.py"]

