FROM docker.1ms.run/nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates cuda-nsight-compute-12-4 \
    && rm -rf /var/lib/apt/lists/*
RUN if ! command -v ncu; then \
    target=$(find /opt/nvidia/nsight-compute -maxdepth 2 -name ncu -type f | sort | tail -1); \
    test -n "$target" && ln -s "$target" /usr/local/bin/ncu; fi
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace
ENTRYPOINT ["python3", "cli.py"]
