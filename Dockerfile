FROM python:3.11-slim-bookworm

# Native ARM64 wheels are available for this TensorFlow release. No x86
# emulation or Mac Python environment is required.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 \
    KERAS_HOME=/workspace/.cache/keras MPLCONFIGDIR=/workspace/.cache/matplotlib \
    TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_experiments.py", "--help"]
