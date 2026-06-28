#!/usr/bin/env python
"""Validation step 3 — is ROCm/PyTorch alive and stable on this machine?

Checks, in order:
  1. torch imports and is a ROCm/HIP build  (torch.version.hip is not None)
  2. torch.cuda.is_available()  (ROCm exposes the AMD GPU through the torch.cuda API)
  3. the device reports as the expected GPU (prints name; warns if it's not a 7900 XTX)
  4. a large matmul runs to completion on-device  (catches the intermittent gfx1100 hangs
     some users hit on rocm/pytorch:latest — if this stalls, pin a known-good image tag)

Run inside the ROCm container (see ../run-container.sh):
    python validation/00_gpu_smoke.py

Exits 0 on PASS, non-zero on FAIL. See CONCEPT.md §7/§9/§10.
"""

import sys
import time


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    print("== OGameSim GPU smoke test (validation step 3) ==\n")

    # 1. torch present and is a ROCm build -------------------------------------------------
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(
            "could not import torch. Run this inside the rocm/pytorch container "
            f"(see ../run-container.sh). Underlying error: {exc!r}"
        )

    print(f"torch version : {torch.__version__}")
    print(f"torch.version.hip  : {torch.version.hip}")
    print(f"torch.version.cuda : {torch.version.cuda}")

    if torch.version.hip is None:
        fail(
            "torch.version.hip is None — this is NOT a ROCm build of torch. A CPU/CUDA wheel was "
            "likely installed over the image's ROCm torch. Check requirements.txt does not pull "
            "torch from PyPI (CONCEPT.md §7, requirements.txt header)."
        )

    # 2. GPU visible to torch --------------------------------------------------------------
    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() is False. The container cannot see the GPU. Check: "
            "(a) `sudo setsebool -P container_use_devices=true` on the host, "
            "(b) --device /dev/kfd --device /dev/dri --group-add keep-groups on `podman run`, "
            "(c) your host user is in the 'render' and 'video' groups. (CONCEPT.md §6)"
        )

    n = torch.cuda.device_count()
    print(f"\ncuda device count : {n}")
    for i in range(n):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    name = torch.cuda.get_device_name(0)
    if "7900" not in name and "gfx1100" not in name.lower():
        print(
            f"\n[WARN] device 0 is '{name}', not obviously a 7900 XTX (gfx1100). "
            "Continuing, but double-check this is the GPU you intend to train on."
        )

    # 3. real compute + stability check ----------------------------------------------------
    # A big matmul exercises rocBLAS the way training will. If gfx1100 is going to hang on this
    # image, this is where it shows up — better here than 6 hours into a run.
    dev = torch.device("cuda")
    size = 8192
    print(f"\nrunning {size}x{size} float32 matmul on {dev} ...", flush=True)
    try:
        a = torch.randn(size, size, device=dev)
        b = torch.randn(size, size, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        fail(
            f"matmul on the GPU raised {exc!r}. If this is a hang/crash rather than an OOM, pin a "
            "known-good rocm/pytorch tag (CONCEPT.md §10) and retry."
        )

    # sanity: result is finite and on-device
    checksum = c.float().sum().item()  # one intentional host read — fine in a smoke test
    if not torch.isfinite(torch.tensor(checksum)):
        fail("matmul produced non-finite values — the GPU/runtime is misbehaving.")

    tflops = (2.0 * size**3) / dt / 1e12
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  done in {dt * 1e3:7.1f} ms  (~{tflops:5.1f} TFLOP/s, checksum {checksum:.3e})")
    print(f"  device memory : {mem_gb:.1f} GB")

    print("\n[PASS] ROCm + PyTorch are alive and stable on this GPU.")
    print("Next: python validation/01_batched_env.py")


if __name__ == "__main__":
    main()
