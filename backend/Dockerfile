# ==============================================================================
# AI-Based Vulnerability Scanner - Backend Dockerfile
# ==============================================================================
# Multi-stage build for optimized production image
# Stage 1: Build dependencies
# Stage 2: Production runtime
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install ML dependencies (optional - larger image)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt || echo "ML dependencies skipped"

# ------------------------------------------------------------------------------
# Stage 2: Production
# ------------------------------------------------------------------------------
FROM python:3.10-slim as production

# Labels
LABEL maintainer="AI Vulnerability Scanner Team"
LABEL version="1.0.0"
LABEL description="AI-Based Vulnerability Scanner Backend"

# Security: Run as non-root user
RUN groupadd -r scanner && useradd -r -g scanner scanner

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=scanner:scanner . .

# Create necessary directories
RUN mkdir -p /app/data/feedback /app/ml/models /app/logs && \
    chown -R scanner:scanner /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=4 \
    DEBUG=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Switch to non-root user
USER scanner

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]

# ==============================================================================
# Build Commands:
# 
# Development:
#   docker build -t vuln-scanner-backend:dev --target builder .
#
# Production:
#   docker build -t vuln-scanner-backend:latest .
#
# Run:
#   docker run -p 8000:8000 -e DEBUG=true vuln-scanner-backend:latest
# ==============================================================================
