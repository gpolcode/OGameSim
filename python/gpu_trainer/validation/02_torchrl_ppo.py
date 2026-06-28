#!/usr/bin/env python
"""Validation step 5 — a REUSED TorchRL PPO loop training a batched, GPU-resident env.

This proves the end goal end-to-end: an established training loop (TorchRL's `ClipPPOLoss` +
`GAE` + `SyncDataCollector`) drives a batched `EnvBase` that lives entirely on the GPU. Nothing
is hand-rolled except the env and the two small networks.

It is ALSO the template that fixes the earlier TorchRL attempt. The two things that stalled it:
  1. an incomplete / mismatched spec set on the EnvBase, and
  2. a `.item()` + Python `if/elif` step that forced host syncs and broke compilation.
Both are done correctly here. `check_env_specs(env)` is called up front to validate (1); the
branchless `_step` handles (2).

Run inside the ROCm container:
    python validation/02_torchrl_ppo.py                                    # GPU-resident PPO
    TORCH_LOGS=graph_breaks python validation/02_torchrl_ppo.py --compile  # audit graph breaks
    python validation/02_torchrl_ppo.py --device cpu                       # logic-only sanity check

Pass criteria (CONCEPT.md §9): ~100 updates run, loss moves, `rocm-smi` shows GPU utilization,
the process is not CPU-bound, and (with --compile) the policy reports zero graph breaks.

Targets torchrl >= 0.6 (with fallbacks for older spec names). If imports of spec classes fail,
it is almost always torchrl<->torch version skew (see requirements.txt), not a logic bug.
"""

import argparse
import sys
import time

import torch
from torch import nn

# --- TorchRL imports, tolerant of the 0.4->0.6 spec rename ----------------------------------
try:
    from torchrl.data import Composite, Unbounded, Categorical
except ImportError:  # older torchrl
    from torchrl.data import (  # type: ignore
        CompositeSpec as Composite,
        UnboundedContinuousTensorSpec as Unbounded,
        DiscreteTensorSpec as Categorical,
    )

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
try:  # torchrl >= 0.13 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector
except ImportError:  # torchrl < 0.13
    from torchrl.collectors import SyncDataCollector as Collector
from torchrl.data.replay_buffers import (
    ReplayBuffer,
    LazyTensorStorage,
    SamplerWithoutReplacement,
)


# ============================================================================================
# The batched, GPU-resident environment (TorchRL EnvBase).
# Same toy economy as 01_batched_env.py, wrapped in the TorchRL contract.
# ============================================================================================
class BatchedResourceEnv(EnvBase):
    batch_locked = True
    N_ACTIONS = 4
    N_RES = 3
    OBS_DIM = 6
    COST = 100.0

    def __init__(self, num_envs: int, max_days: int = 64, device="cpu", seed: int = 0):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.num_envs = num_envs
        self.max_days = max_days
        self._make_specs()
        # persistent batched state, all on-device
        self._state = torch.zeros(num_envs, self.OBS_DIM, device=self.device)
        self._day = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._set_seed(seed)

    # --- specs: get these right and TorchRL "just works"; get them wrong and it stalls -------
    def _make_specs(self) -> None:
        n = self.num_envs
        self.observation_spec = Composite(
            observation=Unbounded(shape=(n, self.OBS_DIM), dtype=torch.float32, device=self.device),
            shape=torch.Size([n]),
            device=self.device,
        )
        # index-based discrete action, one per env -> shape (n,)
        self.action_spec = Categorical(self.N_ACTIONS, shape=torch.Size([n]), device=self.device)
        self.reward_spec = Unbounded(shape=(n, 1), dtype=torch.float32, device=self.device)
        self.done_spec = Composite(
            done=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            terminated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            truncated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            shape=torch.Size([n]),
            device=self.device,
        )

    def _fresh_state(self) -> torch.Tensor:
        s = torch.zeros(self.num_envs, self.OBS_DIM, device=self.device)
        s[:, 3:] = 1.0  # base production
        return s

    def _reset(self, tensordict=None):
        n = self.num_envs
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict.get("_reset").reshape(n)
        else:
            mask = torch.ones(n, dtype=torch.bool, device=self.device)

        fresh = self._fresh_state()
        self._state = torch.where(mask.unsqueeze(1), fresh, self._state)
        self._day = torch.where(mask, torch.zeros_like(self._day), self._day)

        false = torch.zeros(n, 1, dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "observation": self._state.clone(),
                "done": false.clone(),
                "terminated": false.clone(),
                "truncated": false.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict.get("action").reshape(self.num_envs)   # (n,) int64, branchless below
        res = self._state[:, :3]
        prod = self._state[:, 3:]

        is_upgrade = action >= 1
        target = torch.clamp(action - 1, min=0)
        sel = torch.nn.functional.one_hot(target, self.N_RES) * is_upgrade.unsqueeze(1)
        sel = sel.bool()
        affordable = res >= self.COST
        do_upgrade = sel & affordable

        res = res - self.COST * do_upgrade.float()
        prod = prod + do_upgrade.float()
        res = res + prod                                            # production tick
        self._state = torch.cat([res, prod], dim=1)
        self._day = self._day + 1

        attempted = sel.any(dim=1)
        succeeded = do_upgrade.any(dim=1)
        penalty = -0.1 * (attempted & ~succeeded).float()
        reward = (res.sum(dim=1) + penalty).unsqueeze(1)           # (n, 1)

        truncated = (self._day >= self.max_days).unsqueeze(1)      # fixed horizon = time limit
        terminated = torch.zeros_like(truncated)
        done = truncated | terminated

        return TensorDict(
            {
                "observation": self._state.clone(),
                "reward": reward.to(torch.float32),
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        self.rng = torch.manual_seed(seed) if seed is not None else None
        return seed


# ============================================================================================
def build_policy_and_value(env, device):
    obs_dim = env.OBS_DIM
    n_actions = env.N_ACTIONS

    # actor: observation -> logits over actions. The out_key is named "logits" so that
    # ProbabilisticActor calls Categorical(logits=...) by KEYWORD (not positionally as probs).
    actor_net = MLP(
        in_features=obs_dim, out_features=n_actions,
        num_cells=[64, 64], activation_class=nn.Tanh, device=device,
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_net = MLP(
        in_features=obs_dim, out_features=1,
        num_cells=[64, 64], activation_class=nn.Tanh, device=device,
    )
    value = ValueOperator(value_net, in_keys=["observation"])      # out_key "state_value"
    return policy, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=16, help="env steps per collected batch")
    ap.add_argument("--iters", type=int, default=100, help="number of PPO update iterations")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--compile", dest="compile", action="store_true", default=False,
                    help="torch.compile the policy to audit the 'fully compiled' goal "
                         "(off by default; can be finicky on bleeding-edge torch/torchrl)")
    args = ap.parse_args()

    print("== TorchRL PPO on a batched GPU env (validation step 5) ==\n")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[FAIL] --device cuda requested but torch.cuda.is_available() is False. "
              "Run step 3 (00_gpu_smoke.py) first.")
        sys.exit(1)
    device = torch.device(args.device)

    env = BatchedResourceEnv(args.num_envs, device=device)
    print(f"device   : {device}")
    print(f"num_envs : {args.num_envs}   obs_dim: {env.OBS_DIM}   n_actions: {env.N_ACTIONS}")

    # ---- validate the env specs up front: this is what the earlier attempt got wrong --------
    print("\nchecking env specs (check_env_specs) ...", flush=True)
    try:
        check_env_specs(env)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] check_env_specs raised:\n  {exc}\n"
              "The EnvBase specs do not match what _reset/_step actually return. Fix the specs "
              "in _make_specs() or the tensordict keys in _reset/_step. (CONCEPT.md §4.1, §5)")
        sys.exit(1)
    print("  specs OK — env is a valid, batched, on-device TorchRL environment.")

    policy, value = build_policy_and_value(env, device)

    compiled = False
    if args.compile and device.type == "cuda":
        try:
            # Compiling the policy is what makes "graph breaks" meaningful under
            # TORCH_LOGS=graph_breaks. If the env/policy are clean, you should see none.
            policy = torch.compile(policy, mode="reduce-overhead")
            compiled = True
            print("policy   : torch.compile(mode='reduce-overhead') enabled "
                  "(run with TORCH_LOGS=graph_breaks to audit)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] torch.compile failed ({exc!r}); continuing uncompiled. "
                  "GPU-resident training is still validated, just not the compile property.")

    frames_per_batch = args.num_envs * args.rollout
    total_frames = frames_per_batch * args.iters

    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,                    # run + store rollouts on the GPU — no CPU round-trip
    )

    advantage = GAE(gamma=0.99, lmbda=0.95, value_network=value, average_gae=True)
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coef=1.0,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    replay = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch,
    )

    print(f"\ntraining: {args.iters} iters x {args.epochs} epochs, "
          f"{frames_per_batch} frames/batch ({args.num_envs}x{args.rollout})\n")

    first_loss = None
    last_loss = None
    t0 = time.perf_counter()
    for i, data in enumerate(collector):
        for _ in range(args.epochs):
            with torch.no_grad():
                advantage(data)
            replay.extend(data.reshape(-1))
            for _ in range(max(1, frames_per_batch // args.minibatch)):
                sub = replay.sample()
                losses = loss_module(sub)
                loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optim.step()
                optim.zero_grad()

        last_loss = float(loss.detach().item())
        if first_loss is None:
            first_loss = last_loss
        if (i + 1) % 10 == 0 or i == 0:
            with torch.no_grad():
                r = data.get(("next", "reward")).mean().item()
            print(f"  iter {i + 1:3d}/{args.iters}   loss {last_loss:9.3f}   mean reward {r:10.2f}")
        if i + 1 >= args.iters:
            break

    collector.shutdown()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    fps = total_frames / dt
    print(f"\nran {args.iters} PPO iterations in {dt:.1f}s   (~{fps/1e6:.2f} M frames/s)")
    print(f"loss: {first_loss:.3f} -> {last_loss:.3f}")
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak GPU memory: {mem:.2f} GB")

    # PASS = the loop actually trained on-device without crashing. (Loss need not monotonically
    # fall on a toy env, but it should be finite and changing.)
    if last_loss is None or not torch.isfinite(torch.tensor(last_loss)):
        print("\n[FAIL] training did not produce a finite loss.")
        sys.exit(1)

    print("\n[PASS] an established TorchRL PPO loop trained a batched, GPU-resident env.")
    if compiled:
        print("       (compiled: re-run with TORCH_LOGS=graph_breaks and confirm ZERO breaks)")
    print("\nThis is the contract the real OGame env must satisfy. Next: port the C# economy "
          "to a batched tensor env (CONCEPT.md §4.3, §8c).")


if __name__ == "__main__":
    main()
