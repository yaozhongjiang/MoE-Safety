#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v docker >/dev/null
docker info >/dev/null
image=moe-real-channel:cu124-v1
if docker image inspect "$image" >/dev/null 2>&1; then
  echo '[reuse] 本地实验镜像已存在，使用 bash validation/run_cached_image.sh 启动即可。'
  exit 0
fi
# Same CUDA 12.4.1 devel / Ubuntu 22.04 image, pinned to the linux/amd64
# manifest instead of resolving its multi-platform index. Mirror stays 1ms.
exec docker build --platform linux/amd64 --pull=false \
  -f validation/Dockerfile.1ms_amd64 -t "$image" .
