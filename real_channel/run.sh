#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == "--local" ]]; then
  shift
  exec "${PYTHON:-python3}" cli.py "$@"
fi
command -v docker >/dev/null || { echo '需要 Docker Engine 和 NVIDIA Container Toolkit；已有 Python/CUDA/ncu 环境可用 bash run.sh --local。' >&2; exit 1; }
docker info >/dev/null
image=moe-real-channel:cu124-v1
docker build --pull -t "$image" .
mkdir -p artifacts cache
# SYS_ADMIN is required by NVIDIA's restricted GPU performance counter policy.
# Expose only the selected, owned GPU. No host PID namespace or privileged mode.
# WSL's container GPU enumeration is most compatible with --gpus all.
# The doctor refuses to run unless exactly one GPU is visible.
exec docker run --rm --gpus "${GPU_REQUEST:-all}" --cap-add=SYS_ADMIN \
  --shm-size=2g \
  -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
  -v "$PWD:/workspace" "$image" "$@"
