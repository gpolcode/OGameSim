# Working notes for `python/gpu_trainer/` (read me first)

This folder moves OGameSim's RL off its CPU-bound pythonnet/C# env onto a **fully GPU-resident**
training loop on **AMD ROCm** (RX 7900 XTX, Bazzite). Orientation: **`CONCEPT.md`** (why + design),
then **`IMPLEMENTATION.md`** (the current work order: port the C# economy to a batched tensor env).

## Where you're running
You are most likely **inside the `ogamesim-dev` container** (launched via `./run-dev.sh`), which has
ROCm torch + TorchRL + this repo at `/workspace`. The GPU is `cuda:0` (7900 XTX);
`HIP_VISIBLE_DEVICES=0` pins it (a Ryzen iGPU otherwise appears as `cuda:1`).
Sanity check anytime: `python python/gpu_trainer/validation/00_gpu_smoke.py`.

## Validated stack (don't drift from it without re-testing)
- torch **2.10.0+rocm7.2.4** (`torch.version.hip` must be non-`None`), device = `"cuda"`.
- **`torchrl==0.13.2`**, **`tensordict==0.13.0`** (baked into the dev image).

## The env contract (the whole point)
The real OGame env MUST look like `validation/02_torchrl_ppo.py::BatchedResourceEnv`:
- `EnvBase(device="cuda", batch_size=[N])`; all state on-device; specs via `Composite`/`Unbounded`/`Categorical`.
- `_step` is **batched and branchless** — gather + masked `torch.where`/`scatter`. **No `.item()`, no Python `if/elif` on actions.**
- `check_env_specs(env)` must pass; `validation/01_batched_env.py` must stay sync-clean.

## Hard "don'ts"
- **Never `pip install torch`** (a PyPI wheel replaces the ROCm build → `torch.version.hip` becomes `None`). Install other libs under the torch constraint, like `Containerfile` does.
- **Never combine `--ipc=host` with `--shm-size`** (Podman rejects it).
- TorchRL 0.13 renames to remember: `SyncDataCollector`→`Collector`; `entropy_coef`/`critic_coef`→`entropy_coeff`/`critic_coeff`.

## How to run things
- Validation ladder: `python validation/00_gpu_smoke.py`, `01_batched_env.py`, `02_torchrl_ppo.py` (`--compile` for the graph-break audit, `--device cpu` for logic-only checks).
- Parity vs C#: don't run pythonnet here — compare against pre-generated C# fixtures (see `IMPLEMENTATION.md` "Parity test").

## Git
Work on branch `claude/gpu-training-loop-concept-rbitt4` (PR #8). Commit + push as you go.
