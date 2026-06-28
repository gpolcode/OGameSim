# gpu_trainer — fully GPU-resident RL for OGameSim (AMD ROCm / Bazzite)

This folder holds the **concept** and **validation scaffolding** for moving OGameSim off its
CPU-bound, pythonnet/C# environment onto a fully GPU-resident training loop on an AMD RX 7900 XTX.

- **Read [`CONCEPT.md`](./CONCEPT.md) first** — it explains why the current loop is CPU-bound, the
  target architecture (batched on-GPU tensor env + TorchRL), the Bazzite/ROCm runtime, and the
  migration plan.
- The `validation/` scripts are a **ladder** you run *before* the big environment rewrite, to
  prove the AMD/ROCm stack works end-to-end on your hardware.

> Nothing here is the final trainer yet. No game logic has been ported. The point is to de-risk
> the platform first.

## TL;DR — run the validation ladder

On your Bazzite machine (not in CI / not on a CPU-only box):

```bash
# 0. one-time host setup (Fedora Atomic SELinux blocks container device access by default)
sudo setsebool -P container_use_devices=true
groups                                   # confirm 'render' and 'video' are listed

# 1. host sees the GPU
ls -l /dev/kfd /dev/dri/renderD128

# 2. build + enter the container (adds torchrl/rl_games on top of rocm/pytorch)
cd python/gpu_trainer
podman build -t ogamesim-gpu -f Containerfile .
./run-container.sh ogamesim-gpu          # or: ./run-container.sh   (uses rocm/pytorch:latest)

# --- inside the container, from /workspace/python/gpu_trainer ---
rocminfo | grep -i gfx                    # expect gfx1100
rocm-smi                                  # expect the 7900 XTX

python validation/00_gpu_smoke.py        # step 3
python validation/01_batched_env.py      # step 4
python validation/02_torchrl_ppo.py      # step 5  (add --compile for the graph-break audit)
```

Each script prints a clear `PASS` / `FAIL` and exits non-zero on failure. **Stop at the first
failure** and fix it before moving on. See `CONCEPT.md` §9 for the pass criteria and §10 for known
gotchas (e.g. pin a `rocm/pytorch` tag if step 3 hangs).

> Validated on a Bazzite + RX 7900 XTX box (2026-06): the base image ships **torch 2.10.0+rocm7.2.4**;
> steps 3–4 pass; the RL libs are **`torchrl==0.13.2` / `tensordict==0.13.0`**.

## Developing inside the GPU container (implementation phase)

Once the ladder is green, do the actual env port with **Claude Code running inside the container**
(direct GPU + repo access). See `CONCEPT.md` §11.

```bash
cd python/gpu_trainer
podman build -t ogamesim-dev -f Containerfile.dev .
./run-dev.sh            # then run `claude` inside; login persists in ~/.config/ogamesim-claude
```

## Files

| File | Purpose |
|---|---|
| `CONCEPT.md` | The design document — read this first. |
| `Containerfile` | Runtime image: `rocm/pytorch` base + `torchrl`/`tensordict` (torch-constrained install). |
| `Containerfile.dev` | Dev image: the runtime + Node.js + Claude Code, for working *inside* the GPU container. |
| `run-container.sh` | `podman run` wrapper (GPU passthrough) for the validation ladder. |
| `run-dev.sh` | `podman run` wrapper for the dev container, with persisted Claude login. |
| `requirements.txt` | Python deps layered on the base image's ROCm torch (do **not** pip-install torch). |
| `validation/00_gpu_smoke.py` | Is ROCm/PyTorch alive? Device, `torch.version.hip`, a large matmul stability check. |
| `validation/01_batched_env.py` | A tiny batched, branchless, on-GPU env. Proves no per-step host sync. |
| `validation/02_torchrl_ppo.py` | The reusable TorchRL PPO loop on that env. The template the real env must satisfy. |

## Notes

- The CPU-only environment you have today (`../pyTorchPlayer`) is unchanged and still works; this
  folder is additive.
- `01_batched_env.py` and `02_torchrl_ppo.py` accept `--device cpu` so you can sanity-check the
  *logic* on any machine, but the real validation must run on the GPU with `--device cuda`.
