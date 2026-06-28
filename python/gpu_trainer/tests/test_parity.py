"""Differential parity: the batched ``OGameTensorEnv`` vs the scalar reference oracle.

Drives identical action sequences through ``ogame/reference.py`` (faithful C# reward) and
``OGameTensorEnv(reward_mode="ogame")`` and asserts per-step reward, the 125-D observation, and the
action mask all match. Also verifies true batching (independent lanes) and the ``"points"`` reward's
telescoping property. Runs on CPU always and CUDA when available — no .NET involved.
"""

import math
import random

import pytest
import torch

from ogame import reference as ref
from ogame.env import OGameTensorEnv

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference_rollout(seq):
    """Return (init_obs, init_mask, [(reward, obs, mask) per step]) from the scalar reference."""
    renv = ref.ReferenceEnv()
    init_obs = renv.reset()
    init_mask = ref.action_mask(renv.player)
    steps = []
    for a in seq:
        obs, r, _term, _ = renv.step(int(a))
        steps.append((r, obs, ref.action_mask(renv.player)))
    return init_obs, init_mask, steps


def _assert_obs_close(got, exp, ctx):
    g = torch.as_tensor(got, dtype=torch.float64)
    e = torch.as_tensor(exp, dtype=torch.float64)
    # exact for int/LUT-derived values; small rel-tol absorbs the float32 cast on large numbers
    assert torch.allclose(g, e, rtol=1e-3, atol=1e-3), f"{ctx}: max rel err {(g - e).abs().max()}"


@pytest.mark.parametrize("device", DEVICES)
def test_batched_differential(device):
    """Several distinct sequences run as separate lanes of ONE env, each matching its own reference."""
    deterministic = [3, 0, 0, 0, 3, 0, 3, 1, 0, 0, 2, 0, 4, 5, 30, 1, 0, 0, 3, 3, 0, 0, 0, 2, 0, 1, 0]
    seqs = [deterministic * 6]
    for s in range(4):
        rng = random.Random(100 + s)
        seqs.append([rng.randint(0, 62) for _ in range(160)])
    length = min(len(s) for s in seqs)
    seqs = [s[:length] for s in seqs]
    k = len(seqs)

    refs = [_reference_rollout(s) for s in seqs]

    env = OGameTensorEnv(num_envs=k, device=device, reward_mode="ogame")
    td = env.reset()
    for lane in range(k):
        _assert_obs_close(td["observation"][lane].tolist(), refs[lane][0], f"lane{lane} reset obs")
        assert td["action_mask"][lane].tolist() == refs[lane][1], f"lane{lane} reset mask"

    for t in range(length):
        actions = torch.tensor([seqs[lane][t] for lane in range(k)], dtype=torch.int64, device=device)
        td.set("action", actions)
        td = env.step(td)["next"]
        for lane in range(k):
            exp_r, exp_obs, exp_mask = refs[lane][2][t]
            got_r = td["reward"][lane, 0].item()
            assert abs(got_r - exp_r) < 1e-4, f"lane{lane} step{t} reward {got_r} != {exp_r}"
            _assert_obs_close(td["observation"][lane].tolist(), exp_obs, f"lane{lane} step{t} obs")
            assert td["action_mask"][lane].tolist() == exp_mask, f"lane{lane} step{t} mask"


def _greedy_action(player):
    """Reference-driven greedy economy builder (best production-increase per cost), to grow points."""
    obs = ref.update_state(player)
    mask = ref.action_mask(player)
    best, best_ratio = 0, -1.0
    for a in range(3, 63):
        if not mask[a]:
            continue
        pidx, rt = a // 3 - 1, a % 3
        base = 5 + pidx * 6 + rt * 2
        cost, incr = obs[base], obs[base + 1]
        ratio = incr / cost if cost > 0 else 0.0
        if ratio > best_ratio:
            best_ratio, best = ratio, a
    if mask[1] and player.astrophysics.level < player.plasma_technology.level + 2:
        return 1
    return best if best_ratio > 0 else 0


@pytest.mark.parametrize("device", DEVICES)
def test_greedy_trajectory_crosses_exploration_buckets(device):
    """A long economy-growing trajectory crosses 5M-point buckets — exercising exploration claims."""
    renv = ref.ReferenceEnv()
    renv.reset()
    env = OGameTensorEnv(num_envs=1, device=device, reward_mode="ogame")
    td = env.reset()
    big_reward_steps = 0
    for _ in range(8000):
        a = _greedy_action(renv.player)
        obs, r, _term, _ = renv.step(a)
        rmask = ref.action_mask(renv.player)
        td.set("action", torch.tensor([a], dtype=torch.int64, device=device))
        td = env.step(td)["next"]
        assert abs(td["reward"][0, 0].item() - r) < 1e-3
        _assert_obs_close(td["observation"][0].tolist(), obs, "greedy obs")
        assert td["action_mask"][0].tolist() == rmask
        if r > 1.0:  # exploration bonus makes the reward spike well past log10 of a single upgrade
            big_reward_steps += 1
    assert float(renv.player.points) > 5_000_000, "trajectory should cross at least one bucket"
    assert big_reward_steps > 0, "exploration bonus never triggered"


@pytest.mark.parametrize("device", DEVICES)
def test_env_points_match_reference(device):
    """The env's internal `points` accumulator (not in the obs) tracks the reference exactly."""
    rng = random.Random(11)
    seq = [rng.randint(0, 62) for _ in range(800)]
    renv = ref.ReferenceEnv()
    renv.reset()
    env = OGameTensorEnv(num_envs=1, device=device, reward_mode="ogame")
    td = env.reset()
    for a in seq:
        renv.step(a)
        td.set("action", torch.tensor([a], dtype=torch.int64, device=device))
        td = env.step(td)["next"]
    ref_points = float(renv.player.points)
    env_points = env.points[0].item()
    assert abs(env_points - ref_points) <= 1e-3 + 1e-6 * abs(ref_points), \
        f"env points {env_points} != reference {ref_points}"


@pytest.mark.parametrize("device", DEVICES)
def test_points_mode_telescopes_to_final_points(device):
    """sum of per-step `points` rewards (no exploration bonus) == log10(final_points + 1)."""
    rng = random.Random(7)
    seq = [rng.randint(0, 62) for _ in range(600)]
    env = OGameTensorEnv(num_envs=1, device=device, reward_mode="points", exploration_bonus=False)
    td = env.reset()
    total = 0.0
    for a in seq:
        td.set("action", torch.tensor([a], dtype=torch.int64, device=device))
        td = env.step(td)["next"]
        total += td["reward"][0, 0].item()
    expected = math.log10(env.points[0].item() + 1.0)
    assert abs(total - expected) < 1e-4
