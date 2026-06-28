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
- Train: `python train.py` (defaults below). `tensorboard --logdir runs/`.

## Current state / results (`train.py`, `ogame/env.py`)
The economy port + GPU-resident masked-PPO loop are done and tuned. Headline numbers (RX 7900 XTX):

**Performance — fully GPU-resident, host-sync-free hot loop (default on cuda):**
- Custom sync-free GPU rollout (no TorchRL Collector `any_done` host-sync), GPU `randperm`
  minibatcher (no CPU replay sampler), single-pass GAE (`SyncFreeGAE`), compiled PPO update.
- **~400k SPS steady at 16384 envs, GPU 100% util.** `--prove-no-sync` flips on
  `set_sync_debug_mode("error")` and asserts **zero host-device syncs** across collection+GAE+update.
- Update CUDA-graph is **off by default**: on ROCm it HSA-faults once `num_envs × rollout` is large
  (e.g. 8192×32); the compiled-only update is stable and just as fast. Opt in via `--update-cudagraph`.

**Strategy / learning gains:**
- Obs are log1p-compressed + standardized in-env (`log_obs` + `set_obs_norm`) — raw ~3e8 obs into the
  Tanh MLP **NaN-diverge** under compile. The exploration bucket bonus is decoupled from the points
  objective (`exploration_weight`, default 0 in `points` mode); `ogame` mode stays bit-exact.
- Linear entropy + LR anneal. Tuned defaults: **16384 envs, rollout 32, γ=0.9997, entropy 0.05→0.01.**
  More envs *hurt* learning per wall-clock (too few PPO updates) — 16384 is the sweet spot.
- Reaches **~242M points (91% of the 266,316,720.384 hand-crafted `OGameSim.Console` reference) with
  full 20-planet expansion** in a 10-min run. Long-horizon credit (high γ, longer rollout, sustained
  entropy) was the key to escaping the partial-expansion plateau.

**Current objective:** *beat* 266M via intrinsic exploration (the greedy reference never sacrifices
short-term ROI). Success = a deterministic (argmax) eval rollout scoring > 266,316,720.384.

**Invariants when changing the loop:** keep all 87 tests green, keep `ogame` mode bit-exact, and keep
`--prove-no-sync` passing (the training hot loop must touch the CPU for nothing).

## Git
The RL tuning + exploration work lives on `feature/rl-tuning` (off `feature/gpu-resident-ogame-env`).
This container has no push credentials — commit locally; the user pushes (`git push -u origin <branch>`).
