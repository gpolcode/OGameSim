#!/usr/bin/env bash
#
# Launch the GPU dev container (Containerfile.dev) with Claude Code, for the implementation phase.
# Claude runs INSIDE the ROCm container, so it can edit the repo and run training with direct GPU
# access. See CONCEPT.md §11.
#
# Build the image first:
#   podman build -t ogamesim-dev -f Containerfile.dev .
#
# Usage:
#   ./run-dev.sh                 # drop into the container; then run `claude`
#   ./run-dev.sh ogamesim-dev    # explicit image name
#   CLAUDE_CFG=~/somewhere ./run-dev.sh   # override where Claude's login/config is persisted
#
# One-time host setup (if you haven't already, see run-container.sh):
#   sudo setsebool -P container_use_devices=true
#
set -euo pipefail

IMAGE="${1:-${IMAGE:-ogamesim-dev}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Persist Claude Code's login/config OUTSIDE the repo so it survives container restarts and never
# lands in git. Mounted at /claude-config; CLAUDE_CONFIG_DIR points Claude at it.
CLAUDE_CFG="${CLAUDE_CFG:-${HOME}/.config/ogamesim-claude}"
mkdir -p "${CLAUDE_CFG}"

if [[ ! -e /dev/kfd ]]; then
  echo "WARNING: /dev/kfd not found on host — the amdgpu driver may not be loaded." >&2
fi

cat <<EOF
Launching dev container '${IMAGE}'
  repo            -> /workspace
  Claude config   -> ${CLAUDE_CFG}  (persisted across runs)
  GPU             -> /dev/kfd + /dev/dri, pinned to HIP_VISIBLE_DEVICES=0

Inside the container:
  claude                 # first run: device-code login (open the printed URL on the host, paste code)
                         # or:  export ANTHROPIC_API_KEY=...   before running claude
  python python/gpu_trainer/validation/00_gpu_smoke.py   # GPU still visible? (sanity)
EOF

# Note: '--ipc=host' already shares host /dev/shm — do NOT add '--shm-size' (Podman rejects both).
exec podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --ipc=host \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HIP_VISIBLE_DEVICES=0 \
  -e CLAUDE_CONFIG_DIR=/claude-config \
  -v "${CLAUDE_CFG}":/claude-config:Z \
  -v "${REPO_ROOT}":/workspace:Z \
  -w /workspace \
  "${IMAGE}"
