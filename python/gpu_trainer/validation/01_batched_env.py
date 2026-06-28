#!/usr/bin/env python
"""Validation step 4 — a tiny batched, branchless, GPU-resident environment.

This is the *contract* the real OGame env must satisfy (CONCEPT.md §4.1):
  * state is a single (num_envs, F) tensor — all envs step in ONE kernel, no Python loop
  * the step is BRANCHLESS — no `.item()`, no `if/elif` on the action — so it never forces a
    device->host sync and never breaks torch.compile
  * everything stays on the GPU

The key check: we run the stepping loop under `torch.cuda.set_sync_debug_mode("error")`, which
raises if ANY operation synchronizes the host with the device. If this script passes on cuda,
the env genuinely has zero per-step host syncs. (This is exactly what the current
`tensor_game.py::_step` fails, because of its `int(action.item())` + `if/elif`.)

Run inside the ROCm container:
    python validation/01_batched_env.py                 # cuda (default)
    python validation/01_batched_env.py --device cpu    # logic-only sanity check anywhere

Exits 0 on PASS, non-zero on FAIL. See CONCEPT.md §4 / §9.
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F


class BatchedResourceEnv:
    """A minimal batched resource game, in pure torch, fully vectorized and branchless.

    state layout, shape (N, 6):  [res0, res1, res2, prod0, prod1, prod2]
    actions, shape (N,) int64:   0 = wait, 1..3 = upgrade production of resource (action-1)

    Upgrading resource i costs `COST` of resource i and permanently raises prod_i by 1.
    Every day, resources increase by their production. Reward = total resources held, minus a
    small penalty for attempting an unaffordable upgrade. All of this is computed for every env
    at once with gather / masked `torch.where` — never a per-env Python branch.
    """

    N_ACTIONS = 4
    N_RES = 3
    COST = 100.0

    def __init__(self, num_envs: int, max_days: int, device: torch.device):
        self.num_envs = num_envs
        self.max_days = max_days
        self.device = device
        self.state = torch.zeros(num_envs, 6, device=device)
        self.day = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, mask: torch.Tensor) -> None:
        """Branchless (partial) reset: only envs where `mask` is True are reset."""
        fresh = torch.zeros_like(self.state)
        fresh[:, 3:] = 1.0  # base production of 1.0 per resource
        m = mask.unsqueeze(1)
        self.state = torch.where(m, fresh, self.state)
        self.day = torch.where(mask, torch.zeros_like(self.day), self.day)

    def step(self, action: torch.Tensor):
        """Vectorized step for all envs. `action` is (N,) int64. Returns (obs, reward, done)."""
        res = self.state[:, :3]
        prod = self.state[:, 3:]

        is_upgrade = action >= 1                                  # (N,) bool
        target = torch.clamp(action - 1, min=0)                  # (N,) which resource (0..2)
        # one-hot of the targeted resource, zeroed out when the action is "wait"
        sel = F.one_hot(target, self.N_RES) * is_upgrade.unsqueeze(1)   # (N, 3) {0,1}
        sel = sel.bool()

        affordable = res >= self.COST                            # (N, 3) bool
        do_upgrade = sel & affordable                            # (N, 3) bool — actually applied

        res = res - self.COST * do_upgrade.float()
        prod = prod + do_upgrade.float()

        # end-of-day production tick
        res = res + prod
        self.state = torch.cat([res, prod], dim=1)
        self.day = self.day + 1

        # reward: resources held, minus 0.1 for an attempted-but-unaffordable upgrade
        attempted = sel.any(dim=1)
        succeeded = do_upgrade.any(dim=1)
        penalty = -0.1 * (attempted & ~succeeded).float()
        reward = res.sum(dim=1) + penalty                        # (N,)

        done = self.day >= self.max_days                         # (N,) bool
        # auto-reset finished envs, branchlessly, so the loop never needs host-side control flow
        self.reset(done)

        return self.state, reward, done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-days", type=int, default=64)
    args = ap.parse_args()

    print("== Batched branchless GPU env test (validation step 4) ==\n")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[FAIL] --device cuda requested but torch.cuda.is_available() is False. "
              "Run step 3 (00_gpu_smoke.py) first.")
        sys.exit(1)

    device = torch.device(args.device)
    env = BatchedResourceEnv(args.num_envs, args.max_days, device)

    # The env state must live on the target device.
    if env.state.device.type != device.type:
        print(f"[FAIL] env.state is on {env.state.device}, expected {device}.")
        sys.exit(1)
    print(f"device         : {device}")
    print(f"num_envs       : {args.num_envs}")
    print(f"state shape    : {tuple(env.state.shape)}  on {env.state.device}")

    # Warm up (compiles kernels / allocates), then time the loop with NO host syncs allowed.
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(5):
        a = torch.randint(0, env.N_ACTIONS, (args.num_envs,), device=device, generator=gen)
        env.step(a)

    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize()

    # --- the load-bearing check: error out on ANY device->host sync inside the loop ----------
    sync_guarded = on_cuda and hasattr(torch.cuda, "set_sync_debug_mode")
    if sync_guarded:
        torch.cuda.set_sync_debug_mode("error")
    elif on_cuda:
        print("[WARN] torch.cuda.set_sync_debug_mode unavailable in this torch; "
              "cannot hard-prove the no-sync property (older torch).")

    t0 = time.perf_counter()
    try:
        for _ in range(args.steps):
            a = torch.randint(0, env.N_ACTIONS, (args.num_envs,), device=device, generator=gen)
            obs, reward, done = env.step(a)
    except RuntimeError as exc:
        if sync_guarded:
            torch.cuda.set_sync_debug_mode("default")
        print(f"\n[FAIL] a device->host synchronization happened inside the step loop:\n  {exc}\n"
              "Something in step() reads a tensor back to the host (.item()/.cpu()/bool()/"
              "data-dependent indexing). The real env must avoid this too. (CONCEPT.md §4.1)")
        sys.exit(1)
    finally:
        if sync_guarded:
            torch.cuda.set_sync_debug_mode("default")

    if on_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # Only now (outside the timed/guarded region) do we read scalars back for reporting.
    assert obs.device.type == device.type
    sps = args.num_envs * args.steps / dt
    print(f"\nstepped {args.steps} times x {args.num_envs} envs in {dt:.3f}s")
    print(f"throughput     : {sps/1e6:.2f} M env-steps/s")
    print(f"last reward mean: {reward.mean().item():.2f}   done count: {int(done.sum().item())}")

    if sync_guarded:
        print("\n[PASS] env stepped fully on-device with ZERO per-step host syncs.")
    else:
        print("\n[PASS] env logic ran on {}; (no-sync hard-check skipped — see warnings above)."
              .format(device))
    print("Next: TORCH_LOGS=graph_breaks python validation/02_torchrl_ppo.py")


if __name__ == "__main__":
    main()
