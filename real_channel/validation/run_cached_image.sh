#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v docker >/dev/null
docker info >/dev/null
image=moe-real-channel:cu124-v1
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo '[FAILED] 本地没有 moe-real-channel:cu124-v1；请先执行 bash validation/build_image_1ms.sh。' >&2
  exit 2
fi
mkdir -p artifacts cache
# Explicitly prohibit a registry pull. The original experiment CLI and its
# resume/integrity checks are invoked through the existing image entrypoint.
exec docker run --pull=never --rm --init --gpus "${GPU_REQUEST:-all}" \
  --cap-add=SYS_ADMIN --shm-size=2g \
  -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
  -v "$PWD:/workspace" "$image" "$@"
