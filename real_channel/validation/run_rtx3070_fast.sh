#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v docker >/dev/null
docker info >/dev/null
image=moe-real-channel:cu124-v1
# Reuse the already installed image. Build only if it is absent.
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --pull -t "$image" .
fi
mkdir -p artifacts cache
if [[ $# -eq 0 ]]; then set -- launch; fi
exec docker run --rm --init --gpus "${GPU_REQUEST:-all}" --cap-add=SYS_ADMIN \
  --shm-size=2g \
  -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
  -v "$PWD:/workspace" --entrypoint python3 "$image" \
  validation/rtx3070_fast.py "$@"
