# Concept: a fully GPU-resident RL training loop for OGameSim (AMD ROCm / Bazzite)

> Status: **concept + validation scaffolding**. No game logic is ported yet. The goal of this
> document is to (a) explain *why* the current setup is CPU-bound, (b) define the target
> architecture, and (c) give a runnable ladder that proves the AMD/ROCm stack works on your
> machine **before** committing to the large environment rewrite.

---

## 1. Context & goal

OGameSim trains an RL agent to maximize OGame points. Today:

- The economy simulator is **C# / .NET 8** (`csharp/OGameSim.Core`).
- Python calls it **per environment step** through **pythonnet / coreclr**
  (`python/pyTorchPlayer/ogame_env/envs/grid_world.py:41` → `Foo.ApplyAction`,
  `:93` → `Foo.UpdateState`).
- Training is a **hand-rolled CleanRL-style PPO** (`python/pyTorchPlayer/ppo.py`) over **4000**
  copies driven by `gym.vector.SyncVectorEnv`.

What you want:

1. Stop maintaining a bespoke PPO loop — **reuse an established training loop**.
2. Have the loop run **fully on the GPU** (a previous TorchRL attempt stalled).
3. Run on **Bazzite OS** inside a container, on an **AMD Radeon RX 7900 XTX**.

---

## 2. Diagnosis — the bottleneck is the *environment*, not the training loop

This is the single most important point in this document.

In `ppo.py` the rollout does, every step:

```python
next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
```

That one line is the cost:

- `action.cpu().numpy()` — **GPU → CPU → NumPy** copy.
- `envs.step(...)` enters `SyncVectorEnv`, a **host-side Python `for` loop** over 4000 envs.
- Each env calls **C# via pythonnet** (`Foo.ApplyAction`, `Foo.UpdateState`). The .NET CLR runs
  on the **CPU**; pythonnet marshals across an in-process Python↔.NET boundary and has **no GPU
  path**.
- The returned NumPy obs is copied **CPU → GPU** again before the next forward pass.

So per iteration you pay ~`4000 × 25` cross-language, cross-device round-trips in Python. The
observed throughput plateau in `pyTorchPlayer/README.md` (≈60k SPS at 4–8k envs, *falling* to
22k at 64k envs) is the signature of a host-bound stepping loop, not a GPU-bound one.

**Consequence:** swapping the training library changes nothing on its own. SB3, rl_games, and
TorchRL would each be *equally* CPU-bound if pointed at the pythonnet env. And "fully compiled"
is blocked by the same wall — `torch.compile` cannot trace through a pythonnet/C# call (it is a
hard graph break), so compilation also *requires* moving the env onto the GPU.

The repo's own `README.md` todo already names the real fix:
> *"Rewrite die Game Simulation vollständig auf der GPU parallelisiert läuft."*

---

## 3. Target architecture

```
            ┌──────────────────────────────────────────────────────────────┐
            │                         GPU (ROCm / HIP)                       │
            │                                                                │
            │   batched env state         policy + value        PPO loss     │
            │   (num_envs, F) tensor  ─▶  net (MLP)         ─▶  ClipPPOLoss   │
            │        ▲                        │                     │         │
            │        └──────── actions ◀──────┘     gradients ◀─────┘         │
            │                                                                │
            │   everything stays in device memory — no .cpu(), no NumPy,      │
            │   no Python branch per step  →  torch.compile-able end to end   │
            └──────────────────────────────────────────────────────────────┘
```

The whole step → forward → loss cycle lives in device memory. The CPU only orchestrates (kicks
off kernels, reads back scalar metrics occasionally for logging).

---

## 4. The environment (the load-bearing rewrite)

### 4.1 Requirements

- **GPU-native**: all state in tensors on `device='cuda'` (ROCm exposes the AMD GPU through the
  `torch.cuda` API — see §7).
- **Batched**: state shaped `(num_envs, F)`; all `num_envs` step in **one** kernel launch, not a
  Python loop.
- **Branchless**: **no `.item()`** and **no Python `if/elif` on the action**. Both are fatal:
  - `.item()` forces a **device → host sync** every step (serializes the GPU, re-introduces the
    CPU bottleneck).
  - A Python branch on a tensor value is a **`torch.compile` graph break**.

  This is exactly why the current `tensor_game.py::_step` does **not** qualify yet — it does
  `int(tensordict.get("action").item())` followed by `if action == 1: ... elif ...`. That, plus
  its un-batched `(6,)` state, is the same shape of mistake that stalled the earlier TorchRL run.

### 4.2 Build on what already exists

`python/pyTorchPlayer/ogame_env/envs/tensor_game.py` is the right seed:

- It already stores state on `self.device`.
- `batch_plan()` (`tensor_game.py:138`) already shows the **correct** vectorized pattern:

  ```python
  payback   = costs / production                  # (N, 3)
  affordable = resources >= costs                 # (N, 3) bool
  masked    = torch.where(affordable, payback, INF)
  return masked.argmin(dim=1)                      # (N,)  — one decision per env, no Python loop
  ```

The rewrite generalizes that pattern to the *whole* step: keep `action` as a `(num_envs,)`
tensor, **gather** the per-action cost/effect, and apply it with masked `torch.where` / `scatter`
across the full batch.

### 4.3 Parity target — port the real economy from C#

The tensor env must reproduce the C# economy so results are comparable. The spec lives in:

| Concern | C# source | Rule to reproduce |
|---|---|---|
| Metal mine | `Entities/MetalMine.cs` | cost `60·1.5^lvl` metal, `15·1.5^lvl` crystal; base prod `30·24`/day |
| Crystal mine | `Entities/CrystalMine.cs` | analogous geometric cost/prod |
| Deuterium synth | `Entities/DeuteriumSynthesizer.cs` | temperature-dependent production |
| Astrophysics | `Entities/Astrophysics.cs` | cost `4000·1.75^lvl` / `8000·1.75^lvl` / `4000·1.75^lvl`; planet count `ceil(astro/2)+1` |
| Plasma tech | `Entities/PlasmaTechnology.cs` | cost `2000·2^lvl` / `4000·2^lvl` / `1000·2^lvl`; production modifier |
| Resource value (MSE) | `Production/ResourceWeight.cs` | metal 1, crystal 2, deut 3 |
| Reward shaping | `Foo.cs::ApplyAction` | `0.1` proceed-day, `log10(points gained)` + 5M-point-bucket exploration bonus (cap ~25), `-0.1` on an unaffordable action |
| Action set | `Foo.cs` | 63 actions: `0` proceed, `1` astro, `2` plasma, `3..62` = 20 planets × {metal, crystal, deut} |
| Observation | `README.md` ("pyTorchPlayer") + `Foo.cs::UpdateState` | 125-D: player MSE + today's prod, astro cost, plasma cost + Δprod, then 20 × (metal cost, Δ, crystal cost, Δ, deut cost, Δ) |

The single most valuable safety net for this port is a **parity test** (see §8c): drive the same
action sequence through C# `Foo` and through the tensor env, assert points/rewards match within
tolerance.

### 4.4 Backend choice (you said you're open to "something faster")

| Backend | Verdict for this project |
|---|---|
| **PyTorch tensor ops** | **Recommended.** ROCm-native (`device='cuda'`), reuses `tensor_game.py`, and is exactly what the chosen trainer (TorchRL/rl_games) consumes. Lowest variance. |
| **Triton fused kernel** | **Optional upgrade, stays on ROCm.** If profiling later shows the env step is the hot path, fuse it into one Triton kernel (Triton has a ROCm backend). Do this *only after* a PyTorch version works and you have a parity test to compare against. |
| JAX / Brax / Gymnax | **Avoid here.** Would give a fully fused XLA loop, but JAX's ROCm support on gfx1100 is far less mature than PyTorch's — you'd debug the framework's AMD backend instead of your game. |
| NVIDIA Warp | **Not viable.** CUDA-only; no AMD path. |

Recommendation: **PyTorch first**, Triton later if needed. Both keep you inside the proven
torch/ROCm lane.

---

## 5. The training loop (reuse, don't rebuild)

### Primary: **TorchRL** ✅

TorchRL is device-agnostic pure torch: an `EnvBase(device='cuda', batch_size=[num_envs])` keeps
all specs and tensordicts on the GPU, so env + policy + loss never leave the device. It pairs
with `torch.compile(mode="reduce-overhead")` + CUDA graphs for the "fully compiled" goal.

**Your concern was "is there a mature loop I can actually reuse?" — yes.** You don't write the
PPO math; you assemble three batteries-included pieces and (optionally) copy a reference script:

| Piece | What you reuse |
|---|---|
| `torchrl.objectives.ClipPPOLoss` | the PPO clipped objective (ratio clipping, value loss, entropy bonus) |
| `torchrl.objectives.value.GAE` | generalized advantage estimation |
| `torchrl.collectors.SyncDataCollector` | the on-device rollout collector (supports `compile_policy` + `cudagraph_policy`) |
| `sota-implementations/ppo/*` in the TorchRL repo | a **complete, tuned PPO training script** to copy and point at your env |

So the "training loop" is reused; you supply the env and the network. The one honest trade-off vs
a `model.learn()` framework: TorchRL is *assemble-from-modules* (a dozen lines of glue), not a
single call. `validation/02_torchrl_ppo.py` in this folder is that glue, written correctly, and
doubles as the template for the real env.

**Why your earlier TorchRL attempt stalled:** the library was fine — the *env* was incomplete
(`TorchRL/compile.py` calls `OGameBatch()` with no args; `state.py`/`tensor_game.py` use
`.item()` + Python branches and an un-batched state). Fix the env (§4.1) and TorchRL works.

### Fallback: **rl_games**

If the TorchRL glue feels heavy, `rl_games` (the Isaac Gym / Isaac Lab trainer) is more turnkey:
a **YAML config + a runner**, purpose-built for GPU-vectorized batched envs that already return
on-device tensors. Pure torch, so ROCm support comes for free via `device='cuda'`. Decide
TorchRL-vs-rl_games **empirically** at validation step 5 — both are viable; keep rl_games in your
back pocket.

### Ruled out: **Stable-Baselines3**

SB3 is the most familiar option but **cannot meet the "fully on GPU" goal**. Its `VecEnv` API
defines observations/rewards/dones as **NumPy** arrays `(n_envs, …)`, and PPO/A2C are designed to
run primarily on CPU. The policy can be on GPU, but obs/actions are converted to NumPy/CPU **at
the VecEnv boundary every step** — the exact round-trip we are trying to eliminate. Useful only as
a correctness baseline; wrong tool for this objective. (SBX/JAX could be GPU-resident but reopens
the JAX-on-ROCm risk from §4.4.)

---

## 6. Runtime — Bazzite + ROCm via a Podman container (not a VM)

Bazzite is an immutable rpm-ostree (Fedora Atomic / Universal Blue) image: you should **not**
layer the ROCm stack onto the host. The host already provides the `amdgpu` kernel driver and the
`/dev/kfd` + `/dev/dri` device nodes, so a container just needs **device passthrough + group +
SELinux** permissions — no kernel work.

**One-time host setup** (SELinux blocks container device access by default on Fedora Atomic):

```bash
sudo setsebool -P container_use_devices=true
groups   # confirm your user is in 'render' and 'video'
```

**Run** (see `run-container.sh`):

```bash
podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --ipc=host --shm-size 8G \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v "$PWD":/workspace:Z \
  rocm/pytorch:latest
```

- `/dev/kfd` = ROCm/HIP compute interface; `/dev/dri` = render nodes.
- `--group-add keep-groups` preserves the host's `render`/`video` GIDs **inside** the container.
  It needs the **`crun`** runtime (Bazzite's default). With `runc` it silently fails and you get
  permission-denied on `/dev/kfd`.
- `--ipc=host --shm-size 8G` for torch dataloaders; `:Z` relabels the mount for SELinux.

**Why not a VM / VFIO passthrough?** You have a single GPU that also drives your display. Binding
the 7900 XTX (Navi 31) to `vfio-pci` blacks out the host, requires display-manager teardown hooks,
and Navi 31 has a history of reset-bug / black-screen issues in passthrough. VFIO only pays off
with a **second** GPU or a need for full guest-OS isolation — not for bare GPU compute. Use the
container.

---

## 7. ROCm / RX 7900 XTX (gfx1100) facts

- **Officially supported.** gfx1100 / RX 7900 XTX is in the ROCm compatibility matrix, supported
  across the whole **ROCm 6.x line and into 7.x** (7.x is current in 2026; 6.x is fine, just not
  newest). Official support is gated to specific Linux distros, but the **container** sidesteps
  that — it only needs the host `amdgpu` driver.
- **Install.** Easiest: use the `rocm/pytorch` image (PyTorch + ROCm preinstalled — install
  nothing). Otherwise AMD's PyTorch ROCm **pip wheels** (from `pytorch.org` or, AMD-recommended,
  `repo.radeon.com`).
- **`HSA_OVERRIDE_GFX_VERSION` is NOT required.** gfx1100 is natively supported. Guides telling
  you to set `HSA_OVERRIDE_GFX_VERSION=11.0.0` are setting it to the card's **own** value — a
  no-op. Only set it to *spoof* an unsupported card.
- **The device is `"cuda"`.** The ROCm build maps the `torch.cuda` API onto AMD GPUs via HIP. Use
  `device='cuda'` and `torch.cuda.is_available()` exactly as on NVIDIA; confirm you're on a ROCm
  build with `torch.version.hip` being **non-`None`**.

---

## 8. Migration phases

- **(a) Validate the stack** — run the ladder in §9. Don't proceed until step 5 is green.
- **(b) Tiny env + trainer end-to-end** — `validation/01` + `02`: a 4-feature batched env proves
  the TorchRL loop trains on-device with zero graph breaks.
- **(c) Port the full economy** to a batched torch env (§4.3) **with a C# ↔ torch parity test**:
  run identical action sequences through `Foo` and the tensor env, assert points/rewards match.
- **(d) Scale + compile** — push `num_envs` up, profile with `rocm-smi`, enable
  `torch.compile(mode="reduce-overhead")` + CUDA graphs, and only then consider a Triton fused
  step if the env is still the hot path.
- **(e) Port logging/checkpointing** — bring over wandb/tensorboard logging and checkpoint
  save/resume from `ppo.py`.

---

## 9. Validation ladder (run on the Bazzite box, in the container)

Run in order; **stop and fix at the first failure** — do not start the economy port (phase c)
until step 5 passes. Files are in `validation/`.

| # | Command | Pass criteria |
|---|---|---|
| 1 | `ls -l /dev/kfd /dev/dri/renderD128` ; `groups` | devices exist; user in `render`,`video` |
| 2 | build/enter container, then `rocminfo \| grep -i gfx` ; `rocm-smi` | shows `gfx1100` and the 7900 XTX |
| 3 | `python validation/00_gpu_smoke.py` | `cuda` available, `torch.version.hip` non-`None`, correct device name, 8192² matmul completes (catches gfx1100 hangs → pin a tag if it stalls) |
| 4 | `python validation/01_batched_env.py` | env steps fully on `cuda`; **no per-step host sync** (enforced via `torch.cuda.set_sync_debug_mode("error")`) |
| 5 | `TORCH_LOGS=graph_breaks python validation/02_torchrl_ppo.py` | ~100 PPO updates, loss moves, `rocm-smi` shows GPU util, process not CPU-bound, **zero graph breaks** |

Steps 4–5 define the *contract* the real OGame env must satisfy: batched, branchless,
graph-break-free, GPU-resident.

---

## 10. Risks & open items

- **gfx1100 stability on `:latest`** — some users hit intermittent GPU hangs. Step 3 catches it;
  if it stalls, **pin a known-good `rocm/pytorch` tag** and match container ROCm to the host
  kernel/amdgpu.
- **OS rebase desync** — a Bazzite rebase can move the host kernel out of sync with a pinned
  container ROCm (the Fedora 41→42 jump broke some setups). **Re-run steps 2–3 after any major OS
  update.**
- **TorchRL vs rl_games** — decided empirically at step 5; both viable.
- **TorchRL/tensordict version coupling** — `torchrl` and `tensordict` must match the base
  image's torch version (see `requirements.txt`). If `02_torchrl_ppo.py` errors on spec classes,
  it's almost always a version-skew issue, not a logic bug.
- **Parity** — the economy port is only trustworthy with the C# ↔ torch parity test (phase c).
