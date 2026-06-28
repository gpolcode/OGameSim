#!/usr/bin/env python
"""Validation step 6 — the REAL OGame env honours the GPU-resident contract.

Same gate as ``01_batched_env.py`` but for ``ogame.env.OGameTensorEnv``: it must step the whole batch
on-device with **zero per-step host syncs** (enforced via ``torch.cuda.set_sync_debug_mode('error')``)
and its specs must satisfy ``check_env_specs``. This is the go/no-go before training.

Run inside the ROCm container:
    python validation/03_ogame_env.py                 # GPU sync-clean + spec check
    python validation/03_ogame_env.py --device cpu    # logic-only (no sync hard-check)
    python validation/03_ogame_env.py --num-envs 65536 --steps 1000
"""

import argparse
import os
import sys
import time

import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ogame.env import OGameTensorEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--reward-mode", default="points", choices=["points", "ogame"])
    args = ap.parse_args()

    print("== Real OGame batched GPU env test (validation step 6) ==\n")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[FAIL] --device cuda requested but torch.cuda.is_available() is False. "
              "Run validation/00_gpu_smoke.py first.")
        sys.exit(1)

    device = torch.device(args.device)
    env = OGameTensorEnv(args.num_envs, device=device, reward_mode=args.reward_mode)

    if env.resources_mse.device.type != device.type:
        print(f"[FAIL] env state on {env.resources_mse.device}, expected {device}.")
        sys.exit(1)
    print(f"device         : {device}")
    print(f"num_envs       : {args.num_envs}")
    print(f"obs_dim        : {env.OBS_DIM}   n_actions: {env.N_ACTIONS}   reward_mode: {args.reward_mode}")

    # --- spec validation (outside the guarded region; check_env_specs itself reads scalars) -----
    print("\nchecking env specs (check_env_specs) ...", flush=True)
    try:
        check_env_specs(env)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] check_env_specs raised:\n  {exc}")
        sys.exit(1)
    print("  specs OK — valid, batched, on-device TorchRL environment.")

    gen = torch.Generator(device=device).manual_seed(0)

    def rand_td():
        a = torch.randint(0, env.N_ACTIONS, (args.num_envs,), device=device, generator=gen)
        return TensorDict({"action": a}, batch_size=torch.Size([args.num_envs]), device=device)

    # warm up (kernels / allocations)
    for _ in range(5):
        env._step(rand_td())
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize()

    # --- load-bearing check: error on ANY device->host sync inside the step loop -----------------
    sync_guarded = on_cuda and hasattr(torch.cuda, "set_sync_debug_mode")
    if sync_guarded:
        torch.cuda.set_sync_debug_mode("error")
    elif on_cuda:
        print("[WARN] torch.cuda.set_sync_debug_mode unavailable; cannot hard-prove no-sync.")

    out = None
    t0 = time.perf_counter()
    try:
        for _ in range(args.steps):
            out = env._step(rand_td())
    except RuntimeError as exc:
        if sync_guarded:
            torch.cuda.set_sync_debug_mode("default")
        print(f"\n[FAIL] a device->host synchronization happened inside _step:\n  {exc}\n"
              "Something reads a tensor back to the host (.item()/.cpu()/bool()/data-dependent "
              "indexing). The branchless contract is violated. (CONCEPT.md §4.1)")
        sys.exit(1)
    finally:
        if sync_guarded:
            torch.cuda.set_sync_debug_mode("default")

    if on_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    assert out["observation"].device.type == device.type
    sps = args.num_envs * args.steps / dt
    print(f"\nstepped {args.steps} times x {args.num_envs} envs in {dt:.3f}s")
    print(f"throughput     : {sps/1e6:.2f} M env-steps/s")
    print(f"last reward mean: {out['reward'].mean().item():.4f}")

    if sync_guarded:
        print("\n[PASS] OGame env stepped fully on-device with ZERO per-step host syncs.")
    else:
        print(f"\n[PASS] OGame env logic ran on {device} (no-sync hard-check skipped).")


if __name__ == "__main__":
    main()
