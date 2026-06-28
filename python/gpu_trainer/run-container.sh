#!/usr/bin/env bash
#
# Launch a ROCm PyTorch container with the GPU passed through, on Bazzite / Fedora Atomic.
# See CONCEPT.md §6 for what every flag does and why.
#
# Usage:
#   ./run-container.sh                 # uses rocm/pytorch:latest
#   ./run-container.sh ogamesim-gpu    # uses the image you built from Containerfile
#   IMAGE=rocm/pytorch:<pinned-tag> ./run-container.sh
#
# One-time host setup (Fedora Atomic SELinux blocks container device access by default):
#   sudo setsebool -P container_use_devices=true
#   # and make sure your user is in the 'render' and 'video' groups:  groups
#
set -euo pipefail

IMAGE="${1:-${IMAGE:-rocm/pytorch:latest}}"

# Mount the repo root (two levels up from this script) at /workspace so the validation scripts
# and the rest of the project are available inside the container.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Pre-flight warnings (non-fatal) — catch the two most common "GPU not visible" causes early.
if [[ ! -e /dev/kfd ]]; then
  echo "WARNING: /dev/kfd not found on host — the amdgpu driver may not be loaded." >&2
fi
if command -v getsebool >/dev/null 2>&1; then
  if ! getsebool container_use_devices 2>/dev/null | grep -q 'on$'; then
    echo "WARNING: SELinux 'container_use_devices' is off. Run:" >&2
    echo "         sudo setsebool -P container_use_devices=true" >&2
  fi
fi

echo "Launching ${IMAGE} with GPU passthrough; repo mounted at /workspace ..."
# Note: '--ipc=host' already shares the host's /dev/shm, so do NOT add '--shm-size' — Podman
# rejects the two together ("cannot set shmsize when running in the host IPC Namespace").
# 'HIP_VISIBLE_DEVICES=0' pins the discrete 7900 XTX (an iGPU may otherwise show up as cuda:1).
exec podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --ipc=host \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "${REPO_ROOT}":/workspace:Z \
  -w /workspace/python/gpu_trainer \
  "${IMAGE}"
