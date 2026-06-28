# Implementation handoff — port the OGame economy to a batched GPU env

> Read `CONCEPT.md` first (the why + architecture). This file is the **work order** for the
> implementation phase, written to be picked up cold by a fresh Claude Code session running
> **inside the dev container** (`./run-dev.sh`).

## Where we are

The concept phase is **done and validated on the real box** (Bazzite, RX 7900 XTX,
torch 2.10.0+rocm7.2.4, torchrl 0.13.2):

- ✅ ROCm/PyTorch alive and stable (`validation/00_gpu_smoke.py`)
- ✅ batched, branchless, on-GPU env with **zero per-step host sync** (`validation/01_batched_env.py`, ~28 M env-steps/s)
- ✅ an established **TorchRL PPO loop** (`ClipPPOLoss` + `GAE` + `Collector`) training a GPU-resident
  env, `check_env_specs` passing (`validation/02_torchrl_ppo.py`)
- ✅ dev container with Claude Code (`Containerfile.dev` + `run-dev.sh`)

The toy env in `02` is a 6-D / 4-action placeholder. **This phase replaces it with the real OGame
economy** as a batched tensor env, then trains it with the same loop.

## The goal

Reimplement the OGame economy (currently C# in `csharp/OGameSim.Core`, called per-step via
pythonnet in `python/pyTorchPlayer/ogame_env/envs/grid_world.py`) as a **batched, branchless,
GPU-resident TorchRL `EnvBase`**, verified for **parity** against the C# sim, and train it with the
TorchRL PPO loop. Target: throughput far above the ~60k SPS pythonnet baseline, fully on the GPU.

## The env contract (non-negotiable)

`validation/02_torchrl_ppo.py::BatchedResourceEnv` is the **working template** — copy its shape:

- `EnvBase(device="cuda", batch_size=[num_envs])`; all state tensors on-device.
- Specs via `Composite` / `Unbounded` / `Categorical` (torchrl 0.13 names).
- `_reset` / `_step` operate on the **whole batch at once** — gather + masked `torch.where`/`scatter`.
- **No `.item()`, no Python `if/elif` on actions.** `validation/01_batched_env.py` enforces this with
  `torch.cuda.set_sync_debug_mode("error")`; the real env must pass that too.
- `check_env_specs(env)` must succeed.

## The economy to reproduce (source of truth = C#)

Read the C# directly; the parity test (below) is what guarantees correctness, not this summary.
Key files: `csharp/OGameSim.Core/Foo.cs` (entry points), `Entities/Player.cs`, `Entities/*Mine*.cs`,
`Entities/Astrophysics.cs`, `Entities/PlasmaTechnology.cs`, `Production/Resources*.cs`.

**Actions — `Foo.ApplyAction` (63 discrete):** `planetIndex = floor(action/3) - 1`.
- `0` → proceed to next day (reward **+0.1**; **only this action advances a day**).
- `1` → upgrade Astrophysics. `2` → upgrade PlasmaTechnology.
- `3..62` → planet `floor(action/3)-1`, resource `action%3` (`0`=metal, `1`=crystal, `2`=deut).
- If the targeted planet isn't unlocked yet (`planetIndex > planetCount-1`) → **-0.1** penalty.

**Reward on an upgrade:** if affordable, spend → `Upgrade()` → reward = `log10(gainedPoints+1) +
explorationReward`; if unaffordable → **-0.1**. `gainedPoints` = increase in `player.Points` from
spending. **Exploration reward** = one-time bonus per 5,000,000-point bucket, value
`25/60 * bucketIndex` (60 buckets over 300M points), claimed once. *(In C# the claim set is a
global static — for parallel envs make it per-env/per-episode; flag this design choice.)*

**Episode length:** capped at **8000 action-steps** (not days) — see `grid_world.py` `maxSteps`.
`ApplyAction` itself never terminates (its `Terminate()` is dead code).

**Observation — `Foo.UpdateState` (125 doubles, all in metal-equivalent "MSE" =
metal·1 + crystal·2 + deut·3 via `ConvertToMetalValue`):**
- `[0]` resources, `[1]` today's production, `[2]` astro upgrade cost, `[3]` plasma upgrade cost,
  `[4]` plasma upgrade production-delta,
- then **20 planets × 6**: metal cost, metal Δprod/day, crystal cost, crystal Δprod/day, deut cost,
  deut Δprod/day. Unlocked-planet slots are 0. (5 + 20·6 = 125.)

**Subtleties to handle:**
- Planet count = `ceil(astroLevel/2) + 1` (planets unlock as Astrophysics levels up; new planets start at level 0).
- Geometric costs/production (e.g. metal mine cost `60·1.5^lvl` metal, `15·1.5^lvl` crystal; astro `…·1.75^lvl`; plasma `…·2^lvl`) — implement branchlessly with `torch.pow`/`exp`, vectorized over the batch.
- **Numeric precision:** points reach ~3e8 and resources grow large — accumulate points/resources in **float64** (or int64) to avoid float32 drift; observations can be float32.

## Parity test (do this — it's how we trust the port)

Don't run pythonnet inside the GPU container (it needs .NET + the built `Game.dll` + a path fix in
`grid_world.py`). Instead **decouple**:

1. On a machine/container with .NET 8, build the sim (`dotnet publish -c Release` in `csharp/`) and
   write a small driver that runs **fixed action sequences** through `Foo.ApplyAction` /
   `Foo.UpdateState`, dumping `(action, reward, terminated, 125-D obs)` per step to a JSON/NPZ fixture.
2. In the GPU container, unit-test the batched torch env against those fixtures: same action sequence
   ⇒ matching rewards and observations within tolerance (mind float precision).

Keep the fixtures in the repo so the parity test runs without .NET.

## Suggested order

1. **Scaffold** the real env from `BatchedResourceEnv`: state for 20 planets + astro/plasma + points
   + day, obs_dim 125, n_actions 63, branchless `_step`.
2. **Economy**: geometric costs/production, affordability mask, spend+upgrade, plasma/astro effects,
   planet unlocking, day-advance only on action 0, reward shaping, exploration buckets.
3. **Parity test** vs the C# fixtures.
4. **Train**: reuse `02`'s loop with obs_dim=125 / n_actions=63 and bigger nets; pull hyperparameters
   from `python/pyTorchPlayer/ppo.py` (PPO clip/entropy/gamma/lambda, lr, num_envs).
5. **Scale + compile**: raise `num_envs`, profile with `rocm-smi`, enable `--compile` and audit graph
   breaks (`TORCH_LOGS=graph_breaks`), then CUDA graphs; optional Triton fuse of the hot step.
6. **Logging/checkpointing**: port wandb/tensorboard + checkpoint save/resume from `ppo.py`.

## TorchRL 0.13 API notes (so you don't re-stumble)

- Collector: `Collector` (the old `SyncDataCollector` is renamed). `02` imports it resiliently.
- `ClipPPOLoss`: `entropy_coeff` / `critic_coeff` (the `*_coef` spellings were removed in v0.11).
- Specs: `Composite`, `Unbounded`, `Categorical` (from `torchrl.data`).
- Pin `torchrl==0.13.2` / `tensordict==0.13.0`; never pip-install `torch` (kills the ROCm build).

## Definition of done for this phase

- Parity test passes against the C# fixtures.
- PPO trains on the real env **fully on the GPU**, `check_env_specs` green, no per-step host sync,
  throughput >> the pythonnet baseline.
- Stretch: `--compile` graph-break-free; logging + checkpoints ported.
