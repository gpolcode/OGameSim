# Beating the hand-crafted reference — exploration notes

Working log for the goal: **a playthrough scoring > 266,316,720.384 points** (the greedy best-ROI
`OGameSim.Console` reference, ported in `tests/test_integration_console.py`). Read `CLAUDE.md` first
for the perf/loop state; this file is the RL/exploration findings and the plan to continue.

## Status: NOT beaten yet. Reliable performance ≈ 80% of reference; a stable path is identified.

---

## Key findings (this session)

1. **The "242M = 91%" headline was a high-variance fluke.** That number was the *max over 16384
   stochastic episodes* (best-of-batch). The honest metric is the **eval mean** over independent
   full-episode playthroughs (`evaluate()` in `train.py`, logged as `eval/points_mean`): best seen
   ≈ **214M ≈ 80% of reference**, and only while exploration stays high.

2. **Annealing backfires — sustained exploration is required.** With entropy annealed 0.05→0.01 and
   LR→0, the policy *collapsed* from ~214M mean (80%) down to ~31M (12%): sharpening makes it commit
   to a worse strategy. **Use `--no-anneal-entropy`** (keep entropy ~0.05). A *mild* LR decay
   (`--lr-final-frac 0.3`) is fine and helps stability; do **not** anneal LR to 0.

3. **The deterministic (argmax) policy is useless here** — it collapses to repeatedly "proceed"
   (action 0, always valid) and scores ~0. The agent only makes progress by *sampling*. So eval is
   **stochastic** by default; `--eval-deterministic` is opt-in and only meaningful once the policy is
   sharply peaked (which it isn't yet). A truly reproducible >266M run needs the policy to *commit* to
   its good strategy without collapsing — still open (see next steps).

4. **More envs HURT learning per wall-clock.** 65536 was the *worst* (too few PPO updates/hour);
   16384 is the sweet spot. This is about gradient updates, not throughput.

5. **Count-based novelty needs reward normalization or it NaN-crashes.** The raw bonus inflated the
   value target until the critic overshot to inf → NaN logits → the MaskedCategorical finite-check
   aborts the process (`_assert_async_cuda_kernel` → HSA exception). Lowering β only delayed it
   (β0.05→iter2, β0.01→iter50). **Fix (done):** standardize the bonus by a running-RMS EMA + clamp
   (`add_novelty()` in `train.py`). With normalization the run is **stable** — reached astroL 22
   (full expansion), points climbing past 29M mean at iter 90, no crash.

6. **Schedules must anneal over wall-clock for `--max-seconds` runs** (fixed). They were keyed to
   `--iters`, which is set huge for timed runs, so `frac_done≈0` and nothing annealed. Now time-based.

7. **ROCm:** the update CUDA-graph HSA-faults once `num_envs × rollout` is large → off by default
   (`--update-cudagraph` to opt in). The compiled-only update is sync-free and ~400k SPS at 16384.

---

## Recommended config to continue from (stable + best-performing regime)

```
python train.py --intrinsic count --intrinsic-weight 0.02 \
                --no-anneal-entropy --lr-final-frac 0.3 \
                --eval-every 150 --max-seconds 3600
```
Sustained exploration (constant entropy 0.05) + normalized count-based novelty (decaying) + mild LR
decay. This was stable in a 200s smoke (astroL 22, 29M mean climbing) but **a full 1-hr run vs the
reference was not finished** (interrupted by shutdown). That run is the immediate next experiment.

## Next steps to actually beat 266M

1. **Run the stable count config for ≥1 hr** and watch `eval/points_mean` vs 266M. (Interrupted; this
   is step 1.) Sweep `--intrinsic-weight` (0.01–0.05) and `--novelty-day-bucket`.
2. **Make the policy commit without collapsing.** The blocker is high variance / no convergence.
   Ideas: anneal entropy only to a *floor* (e.g. 0.05→0.02, not →0); or a **two-phase** schedule —
   explore (high entropy + novelty) then a short *exploit* phase (novelty off, entropy→0.02, low LR)
   to crystallize the best strategy. Then `--eval-deterministic` may finally be meaningful.
3. **Reward last-mile (deferred by user, revisit if exploration stalls):** `log10(points)` gives
   ~0 marginal reward past ~240M, so surpassing the reference is barely incentivized. A gentler
   potential (e.g. `points**(2/3)`) or a one-time bonus for crossing 266M would help — `ogame` mode
   must stay bit-exact.
4. **RND** (`--intrinsic rnd`) is stubbed in the CLI but **not implemented** — add it as a
   complementary curiosity signal if count-based novelty saturates.
5. **Investigate over-investment:** novelty pushed astroL to 22 (beyond the 20 needed for all planets)
   — the signature may reward useless astro levels. Consider a novelty signature weighted toward
   *production/points* rather than raw astro level.

## Invariants (don't regress)
- All 87 tests green; `ogame` reward mode bit-exact; economy/LUTs/reference untouched.
- `--prove-no-sync` must keep passing (the whole hot loop, incl. novelty `index_add_`/gather, is
  host-sync-free).

## Metrics to watch (TensorBoard, `tensorboard --logdir runs/`)
- `eval/points_mean` ← **the honest headline** (vs 266.3M), `eval/points_max`, `eval/frac_beating_ref`
- `explore/intrinsic_mean`, `explore/beta`, `explore/novel_frac`
- `points/*`, `progress/astro_max`, `perf/sps`

## CLI added this session
`--eval-every/-envs/-steps/--eval-deterministic`; `--intrinsic {none,count,rnd,both}`,
`--intrinsic-weight[-final]`, `--novelty-bits`, `--novelty-day-bucket`; `--max-seconds`;
`--no-anneal-entropy/--no-anneal-lr`, `--lr-final-frac`; `--update-cudagraph`, `--prove-no-sync`.
