#!/usr/bin/env python
"""PPO training on the GPU-resident OGame env — fully on-device, action-masked, TensorBoard-logged.

Reuses TorchRL's standard loop (``Collector`` -> ``GAE`` -> ``ClipPPOLoss`` -> replay minibatches),
exactly like ``validation/02_torchrl_ppo.py``, but drives ``ogame.env.OGameTensorEnv`` (obs_dim 125,
63 discrete actions). The policy samples from a ``MaskedCategorical`` over the env's ``action_mask`` so
invalid actions (unaffordable / locked) are never chosen — no penalty shaping needed. The default
``reward_mode="points"`` makes the discounted return track total points, the objective.

Monitoring is TensorBoard only (no wandb). The headline metric is **final points per episode**
(mean & max); episodes are a synchronized 8000 action-steps.

Run inside the ROCm container:
    python train.py                                            # GPU-resident PPO
    python train.py --num-envs 16384 --iters 2000
    TORCH_LOGS=graph_breaks python train.py --compile          # audit graph breaks
    python train.py --device cpu --num-envs 64 --iters 5       # logic-only smoke
    tensorboard --logdir runs/
"""

import argparse
import os
import sys
import time

import torch
from torch import nn

try:
    from torchrl.data import Composite, Unbounded, Categorical  # noqa: F401
except ImportError:  # pragma: no cover
    pass
from tensordict.nn import TensorDictModule
try:
    from tensordict.nn import CudaGraphModule
except ImportError:  # pragma: no cover
    CudaGraphModule = None
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
try:  # torchrl >= 0.13 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector
except ImportError:  # pragma: no cover
    from torchrl.collectors import SyncDataCollector as Collector
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ogame.env import OGameTensorEnv  # noqa: E402
from tensordict import TensorDict, is_tensor_collection  # noqa: E402

# The hand-crafted greedy best-ROI reference (OGameSim.Console) scores exactly this over 8000 steps;
# mirrors tests/test_integration_console.py::EXPECTED_POINTS. The training goal is to *beat* it.
REFERENCE_POINTS = 266_316_720.384


@torch.no_grad()
def evaluate(eval_env, policy, steps, deterministic=False):
    """Full-episode rollout -> final points per eval env. Sync-free, no grad.

    Default is STOCHASTIC (ExplorationType.RANDOM) — the policy as it actually plays, with each eval
    env an independent playthrough (so max = best-of-batch). The masked-argmax (DETERMINISTIC) mode
    tends to collapse to repeatedly 'proceed' until entropy has annealed low, so it is opt-in."""
    et = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    td = eval_env.reset().select("observation", "action_mask")
    with set_exploration_type(et):
        for _ in range(steps):
            policy(td)
            out = eval_env.step(td)
            td = out.get("next").select("observation", "action_mask")
    return eval_env.points  # (eval_envs,) float64, accumulated over the single episode


class SyncFreeGAE(GAE):
    """GAE without the per-iteration host sync.

    Stock ``_call_value_nets`` runs ``_sanitize_next_obs_nan`` which does ``if not nan_mask.any(): ...``
    — a GPU->CPU sync every iteration. This env never emits NaN next-observations (log1p of
    non-negative magnitudes, then standardized), so the sanitization is a no-op we can skip."""

    def _sanitize_next_obs_nan(self, data, in_keys):
        return data


class GraphSafeClipPPOLoss(ClipPPOLoss):
    """ClipPPOLoss whose entropy path has no host sync, so the whole update is CUDA-graph capturable.

    Stock ``_get_entropy`` does ``if not entropy.isfinite().all(): ...`` — a data-dependent branch that
    is illegal during HIP/CUDA-graph capture. The analytic entropy of a ``MaskedCategorical`` is always
    finite, so we return it directly (same output shaping as the parent)."""

    def _get_entropy(self, dist, adv_shape):
        entropy = dist.entropy()
        if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
            entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)


@torch.no_grad()
def estimate_obs_stats(env, n_steps):
    """Per-dim mean/std of the (log1p-compressed) observation from a short random rollout.

    Run on-device. The scale floor (1.0) keeps rarely-varying dims (e.g. planets that unlock only
    later in training) from exploding once they switch on — log1p obs are already O(1-20)."""
    td = env.reset()
    acc = []
    for _ in range(n_steps):
        td.set("action", torch.randint(0, env.N_ACTIONS, (env.num_envs,), device=env.device))
        td = env.step(td)["next"]
        acc.append(td.get("observation"))
    obs = torch.stack(acc).reshape(-1, env.OBS_DIM)
    return obs.mean(0), obs.std(0).clamp_min(1.0)


def build_policy_and_value(env, device, hidden):
    """Actor over a MaskedCategorical (reads logits + the env's action_mask), and a value head."""
    actor_net = MLP(
        in_features=env.OBS_DIM, out_features=env.N_ACTIONS,
        num_cells=hidden, activation_class=nn.Tanh, device=device,
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        # map distribution kwargs -> tensordict keys: MaskedCategorical(logits=..., mask=action_mask)
        in_keys={"logits": "logits", "mask": "action_mask"},
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    value_net = MLP(
        in_features=env.OBS_DIM, out_features=1,
        num_cells=hidden, activation_class=nn.Tanh, device=device,
    )
    value = ValueOperator(value_net, in_keys=["observation"])
    return policy, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--rollout", type=int, default=32, help="env steps per collected batch")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=125, help="width of the two MLP hidden layers")
    # gamma very near 1: episodes are 8000 steps and the objective is total points; planet/astro
    # expansion only pays off over many future days, so a myopic gamma under-credits it and the agent
    # plateaus at partial expansion. Tuning runs: gamma 0.999 -> 0.9997 lifted final points ~1.6x and
    # took astro to full (20/20 planets). lambda from ppo.py.
    ap.add_argument("--gamma", type=float, default=0.9997)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    # sustained exploration (0.05 -> 0.01) beat 0.03 -> 0 in tuning: keeps probing astro/expansion late.
    ap.add_argument("--entropy-coeff", type=float, default=0.05, help="entropy coeff at start of training")
    ap.add_argument("--critic-coeff", type=float, default=0.5)
    ap.add_argument("--reward-mode", default="points", choices=["points", "ogame"])
    ap.add_argument("--no-exploration-bonus", dest="exploration_bonus", action="store_false", default=True)
    # points mode: the raw bucket bonus (~700/episode) swamps the log10-points base (~8); scale it down
    # (0 = pure telescoping-points objective + policy exploration). ogame mode ignores this (stays exact).
    ap.add_argument("--exploration-weight", type=float, default=0.0)
    # observation conditioning: log1p-compress (raw MSE reach ~3e8) then standardize to ~N(0,1)
    ap.add_argument("--no-log-obs", dest="log_obs", action="store_false", default=True)
    ap.add_argument("--no-obs-norm", dest="obs_norm", action="store_false", default=True)
    ap.add_argument("--obs-norm-steps", type=int, default=200, help="random steps to estimate obs stats")
    # schedules (CleanRL-style linear anneal)
    ap.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false", default=True)
    ap.add_argument("--no-anneal-entropy", dest="anneal_entropy", action="store_false", default=True)
    ap.add_argument("--entropy-final", type=float, default=0.01, help="entropy coeff at end of training")
    ap.add_argument("--lr-final-frac", type=float, default=0.0, help="final LR = lr * this fraction")
    ap.add_argument("--metrics-every", type=int, default=10, help="iters between host-sync metric reads")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="wall-clock cap; 0 = run all --iters")
    # deterministic eval vs the hand-crafted reference (the success metric for beating it)
    ap.add_argument("--eval-every", type=int, default=0, help="iters between deterministic evals; 0=off")
    ap.add_argument("--eval-envs", type=int, default=256,
                    help="parallel envs for the eval rollout (deterministic -> all identical; few suffice)")
    ap.add_argument("--eval-steps", type=int, default=8000, help="steps per eval episode (reference uses 8000)")
    ap.add_argument("--eval-deterministic", action="store_true", default=False,
                    help="eval with masked-argmax instead of sampling (collapses to 'proceed' until entropy is low)")
    # intrinsic exploration: reward visiting under-explored *strategic* configs (astro/plasma/planets/
    # day/points-tier) the greedy reference never reaches. Count-based pseudo-counts: bonus=beta/sqrt(count).
    ap.add_argument("--intrinsic", default="none", choices=["none", "count", "rnd", "both"])
    ap.add_argument("--intrinsic-weight", type=float, default=0.05, help="beta at start of training")
    ap.add_argument("--intrinsic-weight-final", type=float, default=0.0, help="beta at end (decays start->final)")
    ap.add_argument("--novelty-bits", type=int, default=22, help="count-table size = 1<<bits")
    ap.add_argument("--novelty-day-bucket", type=int, default=50, help="days per signature bucket")
    ap.add_argument("--compile", action="store_true", default=False)
    # Fully GPU-resident hot loop (default ON for cuda): a custom sync-free rollout (no collector
    # any_done host-sync) + a compiled PPO update over a GPU randperm minibatcher. Zero CPU compute /
    # zero host-device syncs per iter (verify with --prove-no-sync). Auto-off on cpu. --no-full-compile
    # falls back to the stock TorchRL Collector path.
    ap.add_argument("--no-full-compile", dest="full_compile", action="store_false", default=True)
    # CUDA-graph the PPO update (lowest launch overhead). OFF by default: on ROCm it HSA-faults once the
    # captured allocation (num_envs x rollout) gets large (e.g. 8192 envs x rollout 32), and the
    # compiled-only update is already sync-free and just as fast at scale. Opt in with --update-cudagraph
    # for small num_envs x rollout. --cudagraph-max-envs additionally caps it by num_envs.
    ap.add_argument("--update-cudagraph", dest="update_cudagraph", action="store_true", default=False)
    ap.add_argument("--cudagraph-max-envs", type=int, default=8192,
                    help="also auto-disable the update cudagraph above this num_envs (ROCm stability)")
    # hard proof that the hot loop touches the CPU for nothing: error on ANY host-device sync for a
    # window of steady-state iters (collection + GAE + compiled update). Metrics are muted in-window.
    ap.add_argument("--prove-no-sync", action="store_true", default=False)
    # default mode audits graph breaks cleanly; "reduce-overhead" (CUDA graphs) reuses the policy
    # output buffer and conflicts with the collector unless step boundaries are marked — opt-in only.
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--logdir", default="runs/ogame_ppo")
    ap.add_argument("--ckpt", default="checkpoints/ogame_ppo.pt")
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    print("== PPO on the GPU-resident OGame env ==\n")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[FAIL] --device cuda requested but cuda not available. Run validation/00_gpu_smoke.py.")
        sys.exit(1)
    device = torch.device(args.device)

    raw_env = OGameTensorEnv(
        args.num_envs, device=device,
        reward_mode=args.reward_mode, exploration_bonus=args.exploration_bonus,
        exploration_weight=args.exploration_weight, log_obs=args.log_obs,
    )
    print(f"device   : {device}")
    print(f"num_envs : {args.num_envs}   obs_dim: {raw_env.OBS_DIM}   n_actions: {raw_env.N_ACTIONS}   "
          f"reward_mode: {args.reward_mode}   log_obs: {args.log_obs}   "
          f"explore_w: {raw_env.exploration_weight}")

    # standardize observations to ~N(0,1) so the Tanh MLP isn't saturated (log1p already in env).
    # Folded INTO the env (branchless, on-device) rather than a TorchRL transform, which would sync
    # every step (its `if self.standard_normal:` on a tensor buffer). Estimate stats first, then install.
    if args.obs_norm:
        loc, scale = estimate_obs_stats(raw_env, args.obs_norm_steps)
        raw_env.set_obs_norm(loc, scale)
        print(f"obs_norm : in-env standardization over {args.obs_norm_steps} random steps "
              f"(loc {float(loc.mean()):.2f}, scale {float(scale.mean()):.2f})")
    env = raw_env

    # dedicated eval env (own lanes), same economy + obs-norm, for deterministic eval vs the reference
    eval_env = None
    if args.eval_every and device.type == "cuda":
        eval_env = OGameTensorEnv(
            args.eval_envs, device=device, reward_mode=args.reward_mode,
            exploration_bonus=args.exploration_bonus, exploration_weight=args.exploration_weight,
            log_obs=args.log_obs,
        )
        if args.obs_norm:
            eval_env.set_obs_norm(loc, scale)
        print(f"eval     : every {args.eval_every} iters, {args.eval_envs} envs x {args.eval_steps} steps "
              f"({'deterministic' if args.eval_deterministic else 'stochastic'}) vs reference {REFERENCE_POINTS:.0f}")

    print("\nchecking env specs (check_env_specs) ...", flush=True)
    check_env_specs(env)
    print("  specs OK — valid, batched, on-device TorchRL environment.")

    policy, value = build_policy_and_value(raw_env, device, [args.hidden, args.hidden])

    full_compile = args.full_compile and device.type == "cuda"
    if full_compile and CudaGraphModule is None:
        print("[WARN] --full-compile requested but CudaGraphModule unavailable; falling back.")
        full_compile = False

    frames_per_batch = args.num_envs * args.rollout
    total_frames = frames_per_batch * args.iters
    collector_kwargs = dict(frames_per_batch=frames_per_batch, total_frames=total_frames, device=device)

    # --- collection ---
    # full_compile: a custom GPU-resident rollout (below) that NEVER calls the TorchRL collector's
    # maybe_reset/any_done (a per-step host sync). Otherwise: the stock Collector, optionally compiled.
    collector = None
    if full_compile:
        if args.compile:
            policy = torch.compile(policy, mode=args.compile_mode)
        print(f"collect  : custom sync-free GPU rollout"
              + (f", compiled policy (mode='{args.compile_mode}')" if args.compile else ", eager policy"))
    else:
        if args.compile and device.type == "cuda":
            try:
                policy = torch.compile(policy, mode=args.compile_mode)
                print(f"policy   : torch.compile(mode='{args.compile_mode}') enabled "
                      "(run with TORCH_LOGS=graph_breaks to audit)")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] torch.compile failed ({exc!r}); continuing uncompiled.")
        try:
            collector = Collector(env, policy, auto_register_policy_transforms=False, **collector_kwargs)
        except TypeError:
            collector = Collector(env, policy, **collector_kwargs)

    gae_cls = SyncFreeGAE if device.type == "cuda" else GAE
    advantage = gae_cls(gamma=args.gamma, lmbda=args.gae_lambda, value_network=value, average_gae=True)
    loss_cls = GraphSafeClipPPOLoss if full_compile else ClipPPOLoss
    loss_module = loss_cls(
        actor_network=policy, critic_network=value,
        clip_epsilon=args.clip_epsilon, entropy_bonus=True,
        entropy_coeff=args.entropy_coeff, critic_coeff=args.critic_coeff,
        log_explained_variance=False,
    )
    use_update_cudagraph = full_compile and args.update_cudagraph and args.num_envs <= args.cudagraph_max_envs
    # capturable optimizer is required for the optimizer.step() to be CUDA-graph captured
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr, capturable=use_update_cudagraph)

    minibatch = args.minibatch

    def _ppo_update(sub):
        """One minibatch step: loss -> backward -> clip -> opt.step. All on-GPU, CUDA-graph safe."""
        losses = loss_module(sub)
        loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
        loss.backward()
        nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
        optim.step()
        # set_to_none=False keeps grad buffers at static addresses (required for graph replay)
        optim.zero_grad(set_to_none=not use_update_cudagraph)
        return losses.detach()

    if full_compile:
        update = torch.compile(_ppo_update, mode=args.compile_mode)
        if use_update_cudagraph:
            update = CudaGraphModule(update, warmup=3)  # clones outputs; captures fwd+bwd+opt as one graph
            print("update   : compiled + cudagraph PPO update (capturable Adam, GPU randperm minibatcher)")
        else:
            why = "disabled" if not args.update_cudagraph else f"num_envs>{args.cudagraph_max_envs}"
            print(f"update   : compiled PPO update, no cudagraph ({why}) — sync-free, scales to large num_envs")
    else:
        update = _ppo_update

    start_iter = 0
    best_points = 0.0
    if args.resume and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        loss_module.load_state_dict(ckpt["loss_module"])
        optim.load_state_dict(ckpt["optim"])
        start_iter = ckpt.get("iter", 0)
        best_points = ckpt.get("best_points", 0.0)
        print(f"resumed from {args.ckpt} at iter {start_iter} (best_points {best_points:.0f})")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"\ntraining: {args.iters} iters x {args.epochs} epochs, {frames_per_batch} frames/batch "
          f"({args.num_envs}x{args.rollout})   tensorboard: {args.logdir}\n")

    def save(i):
        torch.save(
            {"loss_module": loss_module.state_dict(), "optim": optim.state_dict(),
             "iter": i, "best_points": best_points},
            args.ckpt,
        )

    def set_entropy_coeff(val):
        """Update the loss module's entropy coeff in-place (tensor buffer or plain attribute)."""
        ec = getattr(loss_module, "entropy_coeff", None)
        if torch.is_tensor(ec):
            ec.fill_(val)
        else:
            loss_module.entropy_coeff = val

    # --- count-based novelty (intrinsic exploration) ----------------------------------------------
    use_count = args.intrinsic in ("count", "both")
    if use_count and not full_compile:
        print("[WARN] --intrinsic needs the custom GPU rollout (full-compile); ignoring on collector path.")
        use_count = False
    novelty_T = 1 << args.novelty_bits
    novelty_counts = torch.zeros(novelty_T, device=device)
    ones_n = torch.ones(args.num_envs, device=device)
    intrinsic_beta = [args.intrinsic_weight]   # mutable holder; updated by the per-iter schedule
    intrinsic_stat = [0.0]                      # last mean bonus, for logging
    D = max(1, args.novelty_day_bucket)

    def strategic_sig():
        """(N,) int64 hash of the current strategic config — astro/plasma/#planets/day-bucket/points-tier.
        Branchless gathers off raw_env state; collisions tolerated via modulo into the count table."""
        astro = raw_env.astro_lvl.clamp(max=63)
        plasma = raw_env.plasma_lvl.clamp(max=63)
        nplan = ((((raw_env.astro_lvl + 1) // 2) + 1).clamp(max=20))
        dayb = (raw_env.day // D).clamp(max=255)
        tier = torch.log10(raw_env.points + 1.0).clamp(min=0, max=10).to(torch.int64)
        h = astro
        h = h * 64 + plasma
        h = h * 21 + nplan
        h = h * 256 + dayb
        h = h * 11 + tier
        return h % novelty_T

    def add_novelty(out):
        """Add beta/sqrt(count+1) to this step's reward and bump the visit counts. Fully sync-free."""
        idx = strategic_sig()
        bonus = intrinsic_beta[0] / torch.sqrt(novelty_counts[idx] + 1.0)
        novelty_counts.index_add_(0, idx, ones_n)
        r = out.get(("next", "reward"))
        r.add_(bonus.reshape_as(r).to(r.dtype))
        intrinsic_stat[0] = bonus.mean()

    @torch.no_grad()
    def gpu_rollout_source():
        """Infinite sync-free GPU rollout. Each yield is one (num_envs, rollout) batch shaped exactly
        like a TorchRL Collector batch. Reset is the env's branchless masked ``_reset`` (a no-op masked
        write when no lane is done) — so there is no ``any_done`` host sync anywhere in collection."""
        cur = env.reset().select("observation", "action_mask")
        while True:
            frames = []
            for _ in range(args.rollout):
                policy(cur)                                  # writes logits/action/sample_log_prob
                out = env.step(cur)                          # adds the "next" sub-tensordict
                if use_count:
                    add_novelty(out)                         # intrinsic bonus into ("next","reward")
                frames.append(out)
                done = out.get(("next", "done")).reshape(env.num_envs)
                reset_in = TensorDict({"_reset": done}, batch_size=env.batch_size, device=device)
                cur = env._reset(reset_in).select("observation", "action_mask")
            yield torch.stack(frames, dim=1)                 # (num_envs, rollout); rollout = time dim

    source = gpu_rollout_source() if full_compile else collector
    minibatch = args.minibatch
    warmup_iters = 5  # exclude compile/cudagraph warmup from the steady-state SPS measurement
    t0 = time.perf_counter()
    t_steady = t0           # reset after warmup so steady SPS excludes one-time compile cost
    frames_steady0 = 0
    last_t, last_frames = t0, 0
    frames_done = 0
    for i, data in enumerate(source, start=start_iter):
        # linear schedules (CleanRL-style): frac_done 0 -> 1 over the run
        frac_done = (i - start_iter) / max(1, args.iters - 1)
        if args.anneal_lr:
            lr_now = args.lr * (1.0 + (args.lr_final_frac - 1.0) * frac_done)
            for g in optim.param_groups:
                g["lr"] = lr_now
        if args.anneal_entropy:
            set_entropy_coeff(args.entropy_coeff + (args.entropy_final - args.entropy_coeff) * frac_done)
        # intrinsic weight decays start -> final so the agent explores early, exploits late
        intrinsic_beta[0] = args.intrinsic_weight + (args.intrinsic_weight_final - args.intrinsic_weight) * frac_done

        # sync-free proof window: error on any host-device sync across collection + GAE + update
        rel = i - start_iter
        if args.prove_no_sync and rel == warmup_iters:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            print(f"[sync-check] host-sync error mode ON at iter {i + 1} (collection + GAE + update)")
        if args.prove_no_sync and rel == warmup_iters + 5:
            torch.cuda.set_sync_debug_mode("default")
            print(f"[sync-check] PASSED — 5 steady-state iters ran with ZERO host-device syncs")

        # GAE ONCE per batch (was redundantly redone every epoch). Then a fully GPU-resident minibatcher:
        # torch.randperm + index_select on-device (no CPU replay sampler), reshuffled per epoch.
        with torch.no_grad():
            advantage(data)
        flat = data.reshape(-1)
        nframes = flat.shape[0]
        usable = (nframes // minibatch) * minibatch  # drop a ragged tail so minibatch shape is static
        for _ in range(args.epochs):
            perm = torch.randperm(nframes, device=device)[:usable].view(-1, minibatch)
            for mb in range(perm.shape[0]):
                losses = update(flat[perm[mb]])

        frames_done += frames_per_batch
        now = time.perf_counter()
        if i - start_iter == warmup_iters:  # start the steady-state clock once graphs are hot
            t_steady, frames_steady0 = now, frames_done
        sps = frames_done / (now - t0)  # cumulative (incl. warmup); no host sync
        steady_sps = (frames_done - frames_steady0) / (now - t_steady) if now > t_steady else sps

        # --- metrics: gated so the per-iter hot path stays sync-free (headline: points/episode) ---
        in_sync_window = args.prove_no_sync and warmup_iters <= rel < warmup_iters + 5
        log_now = ((i + 1) % args.metrics_every == 0 or i == start_iter
                   or i + 1 >= start_iter + args.iters) and not in_sync_window
        if log_now:
            with torch.no_grad():
                points = raw_env.points
                cur_max = points.max().item()
                best_points = max(best_points, cur_max)
                mean_reward = data.get(("next", "reward")).mean().item()
                writer.add_scalar("reward/mean", mean_reward, i)
                writer.add_scalar("points/mean", points.mean().item(), i)
                writer.add_scalar("points/max", cur_max, i)
                writer.add_scalar("points/best_so_far", best_points, i)
                writer.add_scalar("progress/astro_max", raw_env.astro_lvl.max().item(), i)
                writer.add_scalar("progress/plasma_max", raw_env.plasma_lvl.max().item(), i)
                writer.add_scalar("progress/day_mean", raw_env.day.float().mean().item(), i)
                writer.add_scalar("loss/objective", losses["loss_objective"].item(), i)
                writer.add_scalar("loss/critic", losses["loss_critic"].item(), i)
                writer.add_scalar("loss/entropy", losses["loss_entropy"].item(), i)
                inst_sps = (frames_done - last_frames) / max(1e-9, now - last_t)
                last_t, last_frames = now, frames_done
                writer.add_scalar("perf/sps", sps, i)
                writer.add_scalar("perf/sps_steady", steady_sps, i)
                writer.add_scalar("perf/sps_inst", inst_sps, i)
                writer.add_scalar("sched/lr", optim.param_groups[0]["lr"], i)
                if use_count:
                    writer.add_scalar("explore/intrinsic_mean", float(intrinsic_stat[0]), i)
                    writer.add_scalar("explore/beta", intrinsic_beta[0], i)
                    writer.add_scalar("explore/novel_frac", float((novelty_counts == 0).float().mean()), i)
                print(f"  iter {i + 1:5d}   reward {mean_reward: .4f}   points mean {points.mean().item():10.0f} "
                      f"max {cur_max:12.0f}   astroL {raw_env.astro_lvl.max().item():2d}   "
                      f"{inst_sps/1e3:6.1f}k SPS (steady {steady_sps/1e3:5.0f}k)")
        # deterministic eval vs the reference (outside the sync-free window; reading points syncs)
        if (eval_env is not None and (i + 1) % args.eval_every == 0 and not in_sync_window):
            ep = evaluate(eval_env, policy, args.eval_steps, deterministic=args.eval_deterministic)
            ev_max, ev_mean = ep.max().item(), ep.mean().item()
            frac = (ep > REFERENCE_POINTS).float().mean().item()
            writer.add_scalar("eval/points_max", ev_max, i)
            writer.add_scalar("eval/points_mean", ev_mean, i)
            writer.add_scalar("eval/frac_beating_ref", frac, i)
            writer.add_scalar("eval/ratio_to_ref", ev_max / REFERENCE_POINTS, i)
            kind = "det" if args.eval_deterministic else "sto"
            beat = "  ** BEATS REFERENCE **" if ev_max > REFERENCE_POINTS else ""
            print(f"  [eval {i + 1:5d}] {kind}. points mean {ev_mean:12.0f} max {ev_max:12.0f}  "
                  f"({100 * ev_max / REFERENCE_POINTS:5.1f}% of ref, {100 * frac:4.1f}% beat){beat}")
        if (i + 1) % args.save_every == 0:
            save(i + 1)
        if i + 1 >= start_iter + args.iters:
            break
        if args.max_seconds and (now - t0) > args.max_seconds:
            print(f"  [max-seconds {args.max_seconds:.0f}s reached at iter {i + 1}]")
            break

    if collector is not None:
        collector.shutdown()
    save(start_iter + args.iters)
    writer.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    dt = end - t0
    steady = (frames_done - frames_steady0) / (end - t_steady) if end > t_steady else frames_done / dt
    print(f"\n[done] {frames_done} frames in {dt:.1f}s  (cumulative {frames_done/dt/1e3:.1f}k SPS, "
          f"steady-state {steady/1e3:.1f}k SPS)  best points {best_points:.0f}")
    print(f"checkpoint: {args.ckpt}   tensorboard: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
