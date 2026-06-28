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
from torchrl.envs.utils import ExplorationType, check_env_specs
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
try:  # torchrl >= 0.13 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector
except ImportError:  # pragma: no cover
    from torchrl.collectors import SyncDataCollector as Collector
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ogame.env import OGameTensorEnv  # noqa: E402


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
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--rollout", type=int, default=16, help="env steps per collected batch")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=125, help="width of the two MLP hidden layers")
    # gamma defaults near 1: the episode is 8000 steps and the objective is total points, so a myopic
    # gamma (ppo.py used 0.95) would under-credit early economy building. lambda from ppo.py.
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--entropy-coeff", type=float, default=0.01)
    ap.add_argument("--critic-coeff", type=float, default=0.5)
    ap.add_argument("--reward-mode", default="points", choices=["points", "ogame"])
    ap.add_argument("--no-exploration-bonus", dest="exploration_bonus", action="store_false", default=True)
    ap.add_argument("--compile", action="store_true", default=False)
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

    env = OGameTensorEnv(
        args.num_envs, device=device,
        reward_mode=args.reward_mode, exploration_bonus=args.exploration_bonus,
    )
    print(f"device   : {device}")
    print(f"num_envs : {args.num_envs}   obs_dim: {env.OBS_DIM}   n_actions: {env.N_ACTIONS}   "
          f"reward_mode: {args.reward_mode}")

    print("\nchecking env specs (check_env_specs) ...", flush=True)
    check_env_specs(env)
    print("  specs OK — valid, batched, on-device TorchRL environment.")

    policy, value = build_policy_and_value(env, device, [args.hidden, args.hidden])

    if args.compile and device.type == "cuda":
        try:
            policy = torch.compile(policy, mode=args.compile_mode)
            print(f"policy   : torch.compile(mode='{args.compile_mode}') enabled "
                  "(run with TORCH_LOGS=graph_breaks to audit)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] torch.compile failed ({exc!r}); continuing uncompiled.")

    frames_per_batch = args.num_envs * args.rollout
    total_frames = frames_per_batch * args.iters
    collector_kwargs = dict(frames_per_batch=frames_per_batch, total_frames=total_frames, device=device)
    try:
        collector = Collector(env, policy, auto_register_policy_transforms=False, **collector_kwargs)
    except TypeError:
        collector = Collector(env, policy, **collector_kwargs)

    advantage = GAE(gamma=args.gamma, lmbda=args.gae_lambda, value_network=value, average_gae=True)
    loss_module = ClipPPOLoss(
        actor_network=policy, critic_network=value,
        clip_epsilon=args.clip_epsilon, entropy_bonus=True,
        entropy_coeff=args.entropy_coeff, critic_coeff=args.critic_coeff,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)
    replay = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(), batch_size=args.minibatch,
    )

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

    t0 = time.perf_counter()
    frames_done = 0
    for i, data in enumerate(collector, start=start_iter):
        for _ in range(args.epochs):
            with torch.no_grad():
                advantage(data)
            replay.extend(data.reshape(-1))
            for _ in range(max(1, frames_per_batch // args.minibatch)):
                sub = replay.sample()
                losses = loss_module(sub)
                loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optim.step()
                optim.zero_grad()

        frames_done += frames_per_batch
        sps = frames_done / (time.perf_counter() - t0)

        # --- metrics (headline: final points per episode) ---
        with torch.no_grad():
            points = env.points
            cur_max = points.max().item()
            best_points = max(best_points, cur_max)
            mean_reward = data.get(("next", "reward")).mean().item()
            writer.add_scalar("reward/mean", mean_reward, i)
            writer.add_scalar("points/mean", points.mean().item(), i)
            writer.add_scalar("points/max", cur_max, i)
            writer.add_scalar("points/best_so_far", best_points, i)
            writer.add_scalar("progress/astro_max", env.astro_lvl.max().item(), i)
            writer.add_scalar("progress/plasma_max", env.plasma_lvl.max().item(), i)
            writer.add_scalar("progress/day_mean", env.day.float().mean().item(), i)
            writer.add_scalar("loss/objective", losses["loss_objective"].item(), i)
            writer.add_scalar("loss/critic", losses["loss_critic"].item(), i)
            writer.add_scalar("loss/entropy", losses["loss_entropy"].item(), i)
            writer.add_scalar("loss/grad_norm", float(grad_norm), i)
            writer.add_scalar("perf/sps", sps, i)

        if (i + 1) % 10 == 0 or i == start_iter:
            print(f"  iter {i + 1:5d}   reward {mean_reward: .4f}   points mean {points.mean().item():10.0f} "
                  f"max {cur_max:12.0f}   astroL {env.astro_lvl.max().item():2d}   {sps/1e3:6.1f}k SPS")
        if (i + 1) % args.save_every == 0:
            save(i + 1)
        if i + 1 >= start_iter + args.iters:
            break

    collector.shutdown()
    save(start_iter + args.iters)
    writer.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"\n[done] {frames_done} frames in {dt:.1f}s  ({frames_done/dt/1e3:.1f}k SPS)  "
          f"best points {best_points:.0f}")
    print(f"checkpoint: {args.ckpt}   tensorboard: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
