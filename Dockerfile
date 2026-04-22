# Multi-stage Dockerfile for DREAMPlace_MetaOpt
# Stage 1: Python 3.11 slim with optimized NumPy/SciPy
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies for NumPy/SciPy compilation and MKL optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    liblapack-dev \
    liblapack3 \
    libopenblas-dev \
    libgomp1 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with optimization flags
# Use pre-built wheels where available for speed
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0 \
    matplotlib>=3.5.0 \
    numba>=0.56.0 \
    Pillow>=9.0.0

# Set environment variables for optimal NumPy/SciPy performance
# Use all available threads (user can override with docker run -e)
ENV OPENBLAS_NUM_THREADS=0 \
    MKL_NUM_THREADS=0 \
    OMP_NUM_THREADS=0 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create cache directory for Numba
RUN mkdir -p /tmp/numba_cache

# Copy entire application
COPY . .

# Expose port for future web GUI if needed
EXPOSE 8000

# Default: batch mode without GUI, using 4 workers
# Override with: docker run -e N_JOBS=8 dreamplace:latest ...
ENV N_JOBS=4 \
    BATCH_MODE=true

# Entry point: run.py in batch mode
ENTRYPOINT ["python", "run.py"]
CMD ["--batch", "--help"]

# Stage 2: GPU variant (optional, built separately)
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 as gpu

WORKDIR /app

# Install Python 3.11 on CUDA base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    gfortran \
    liblapack-dev \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY requirements.txt .

# Install dependencies + GPU-specific packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0 \
    matplotlib>=3.5.0 \
    numba>=0.56.0 \
    cupy-cuda12x>=12.0.0 \
    Pillow>=9.0.0

ENV OPENBLAS_NUM_THREADS=0 \
    MKL_NUM_THREADS=0 \
    OMP_NUM_THREADS=0 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    N_JOBS=4 \
    BATCH_MODE=true \
    USE_GPU=true

RUN mkdir -p /tmp/numba_cache

COPY . .

EXPOSE 8000

ENTRYPOINT ["python", "run.py"]
CMD ["--batch", "--help"]
