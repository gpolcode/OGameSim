"""``OGameTensorEnv`` — the batched, branchless, GPU-resident OGameSim economy (TorchRL ``EnvBase``).

Satisfies the same contract as ``validation/02_torchrl_ppo.py::BatchedResourceEnv``: all state lives on
the device, ``_step`` is fully vectorised (gather / masked ``where`` / ``scatter`` — no ``.item()``, no
Python ``if/elif`` on actions), and ``check_env_specs`` passes. The economy exactly mirrors the C#
``OGameSim.Core`` (validated against ``ogame/reference.py`` and the ported unit tests); costs and
production come from precomputed lookup tables (``ogame/luts.py``) so the hot loop has no ``pow``.

Two reward modes (economy + observations identical in both):
- ``"points"`` (default, training): potential-based ``log10(points+1)`` deltas → the discounted return
  maximises total points directly. Pairs with action masking (the emitted ``action_mask``).
- ``"ogame"`` (parity): the exact C# reward (``+0.1`` proceed / ``log10(gain+1)+exploration`` / ``-0.1``).
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

try:
    from torchrl.data import Composite, Unbounded, Categorical
except ImportError:  # older torchrl
    from torchrl.data import (  # type: ignore
        CompositeSpec as Composite,
        UnboundedContinuousTensorSpec as Unbounded,
        DiscreteTensorSpec as Categorical,
    )
from tensordict import TensorDict
from torchrl.envs import EnvBase

from .luts import build_luts

N_ACTIONS = 63
OBS_DIM = 125
MAX_PLANETS = 20
N_BUCKETS = 60
REWARD_DISTRIBUTION = 5_000_000.0
EXPLORATION_MAX_VALUE = 25.0


class OGameTensorEnv(EnvBase):
    batch_locked = True
    N_ACTIONS = N_ACTIONS
    OBS_DIM = OBS_DIM

    def __init__(
        self,
        num_envs: int,
        device="cpu",
        max_steps: int = 8000,
        reward_mode: str = "points",
        exploration_bonus: bool = True,
        exploration_weight: float = 1.0,
        log_obs: bool = False,
        seed: int = 0,
    ):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        assert reward_mode in ("points", "ogame")
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        # Exploration-bucket bonus weight. In faithful "ogame" mode it is the exact C# term
        # (added unscaled). In "points" mode the raw bonus (~700 summed over an episode) dwarfs the
        # telescoping log10-points base (~8), so it is *scaled* by ``exploration_weight`` — set small
        # (or 0) so the smooth total-points potential stays the real objective.
        if reward_mode == "ogame":
            self.exploration_weight = 1.0
        else:
            self.exploration_weight = float(exploration_weight) if exploration_bonus else 0.0
        # claim bookkeeping is only needed when the bonus actually contributes
        self.include_exploration = self.exploration_weight != 0.0
        # log1p-compress observations (raw MSE magnitudes reach ~3e8; a Tanh MLP saturates otherwise).
        # Off by default so the faithful observation parity tests still see raw values.
        self.log_obs = log_obs
        # optional affine obs standardization, applied branchlessly in-env (no TorchRL transform, which
        # would sync every step). Set via ``set_obs_norm`` once stats are estimated.
        self._obs_loc = None
        self._obs_scale = None

        self.luts = build_luts(self.device)
        self._lmax = self.luts.lmax
        self._ar = torch.arange(num_envs, device=self.device)
        self._planet_ids = torch.arange(MAX_PLANETS, device=self.device)

        self._make_specs()
        self._init_state()
        self._set_seed(seed)

    # --- specs ----------------------------------------------------------------------------------
    def _make_specs(self) -> None:
        n = self.num_envs
        self.observation_spec = Composite(
            observation=Unbounded(shape=(n, OBS_DIM), dtype=torch.float32, device=self.device),
            action_mask=Categorical(2, shape=(n, N_ACTIONS), dtype=torch.bool, device=self.device),
            shape=torch.Size([n]),
            device=self.device,
        )
        self.action_spec = Categorical(N_ACTIONS, shape=torch.Size([n]), device=self.device)
        self.reward_spec = Unbounded(shape=(n, 1), dtype=torch.float32, device=self.device)
        self.done_spec = Composite(
            done=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            terminated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            truncated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=self.device),
            shape=torch.Size([n]),
            device=self.device,
        )

    # --- state ----------------------------------------------------------------------------------
    def _init_state(self) -> None:
        n = self.num_envs
        dev = self.device
        self.metal_lvl = torch.zeros(n, MAX_PLANETS, dtype=torch.int64, device=dev)
        self.crystal_lvl = torch.zeros(n, MAX_PLANETS, dtype=torch.int64, device=dev)
        self.deut_lvl = torch.zeros(n, MAX_PLANETS, dtype=torch.int64, device=dev)
        self.astro_lvl = torch.zeros(n, dtype=torch.int64, device=dev)
        self.plasma_lvl = torch.zeros(n, dtype=torch.int64, device=dev)
        self.resources_mse = torch.zeros(n, dtype=torch.int64, device=dev)
        self.points = torch.zeros(n, dtype=torch.float64, device=dev)
        self.day = torch.zeros(n, dtype=torch.int64, device=dev)
        self.step_ctr = torch.zeros(n, dtype=torch.int64, device=dev)
        self.claimed = torch.zeros(n, N_BUCKETS, dtype=torch.bool, device=dev)

    def _set_seed(self, seed):
        self.rng = torch.manual_seed(seed) if seed is not None else None
        return seed

    def set_obs_norm(self, loc, scale):
        """Install a fixed affine obs standardization (obs-loc)/scale, applied in-env in ``_observe``.

        Branchless and device-resident — unlike a TorchRL ``ObservationNorm`` transform whose
        ``if self.standard_normal:`` (a tensor buffer) triggers a host sync on every step."""
        self._obs_loc = loc.to(self.device, torch.float32)
        self._obs_scale = scale.to(self.device, torch.float32)

    # --- derived quantities (branchless gathers) ------------------------------------------------
    def _planet_active(self):
        """(N,20) bool — planet p unlocked iff p < ceil(astro/2)+1 (capped at 20)."""
        planet_count = (((self.astro_lvl + 1) // 2) + 1).clamp(max=MAX_PLANETS)  # (N,)
        return self._planet_ids.unsqueeze(0) < planet_count.unsqueeze(1)

    def _production(self, active):
        """Today's full production (mine + plasma modifier), integer-exact. Returns MSE + components."""
        ml = self.metal_lvl.clamp(max=self._lmax - 1)
        cl = self.crystal_lvl.clamp(max=self._lmax - 1)
        dl = self.deut_lvl.clamp(max=self._lmax - 1)
        mineM = (self.luts.metal_prod[ml] * active).sum(dim=1)      # (N,) int64
        mineC = (self.luts.crystal_prod[cl] * active).sum(dim=1)
        mineD = (self.luts.deut_prod[dl] * active).sum(dim=1)
        pl = self.plasma_lvl
        # Resources * ResourcesModifier floors each component (Resources.cs:13); exact via int floordiv.
        totM = mineM + (mineM * pl) // 100
        totC = mineC + (mineC * pl * 66) // 10000
        totD = mineD + (mineD * pl * 33) // 10000
        production_mse = totM + 2 * totC + 3 * totD
        return production_mse, totM, totC, totD

    def _cost_gathers(self):
        """Per-planet upgrade cost (MSE, float64) and increase-per-day (MSE) for all three mines."""
        ml = self.metal_lvl.clamp(max=self._lmax - 1)
        cl = self.crystal_lvl.clamp(max=self._lmax - 1)
        dl = self.deut_lvl.clamp(max=self._lmax - 1)
        return {
            "metal_cost": self.luts.metal_cost_mse[ml],     # (N,20) f64
            "crystal_cost": self.luts.crystal_cost_mse[cl],
            "deut_cost": self.luts.deut_cost_mse[dl],
            "metal_incr": self.luts.metal_incr_mse[ml],     # (N,20) i64
            "crystal_incr": self.luts.crystal_incr_mse[cl],
            "deut_incr": self.luts.deut_incr_mse[dl],
        }

    def _observe(self, production_mse, totM, totC, totD, active):
        """Assemble the (N,125) observation and the (N,63) action mask from current state."""
        n = self.num_envs
        g = self._cost_gathers()

        # --- observation (Foo.UpdateState) ---
        # plasma upgrade production-delta: production * (upgradedMod - mod), modifier diff = (1/100, .66/100, .33/100)
        plasma_delta = (totM // 100) + 2 * ((totC * 66) // 10000) + 3 * ((totD * 33) // 10000)
        astro_cost = self.luts.astro_cost_mse[self.astro_lvl.clamp(max=self._lmax - 1)]
        plasma_cost = self.luts.plasma_cost_mse[self.plasma_lvl.clamp(max=self._lmax - 1)]
        head = torch.stack(
            [
                self.resources_mse.double(),
                production_mse.double(),
                astro_cost,
                plasma_cost,
                plasma_delta.double(),
            ],
            dim=1,
        )  # (N,5)
        body = torch.stack(
            [
                g["metal_cost"], g["metal_incr"].double(),
                g["crystal_cost"], g["crystal_incr"].double(),
                g["deut_cost"], g["deut_incr"].double(),
            ],
            dim=2,
        )  # (N,20,6)
        body = body * active.unsqueeze(2)  # inactive planet slots -> 0
        obs = torch.cat([head, body.reshape(n, MAX_PLANETS * 6)], dim=1)
        if self.log_obs:
            # all components are non-negative MSE magnitudes; log1p is monotonic and well-conditioned
            obs = torch.log1p(obs.clamp(min=0))
        obs = obs.to(torch.float32)
        if self._obs_loc is not None:
            obs = (obs - self._obs_loc) / self._obs_scale

        # --- action mask ---
        res = self.resources_mse.unsqueeze(1)  # (N,1) int64, promotes vs f64 cost
        mine_cost = torch.stack(
            [g["metal_cost"], g["crystal_cost"], g["deut_cost"]], dim=2
        ).reshape(n, MAX_PLANETS * 3)  # (N,60) in action order (planet, resType)
        active_60 = active.unsqueeze(2).expand(n, MAX_PLANETS, 3).reshape(n, MAX_PLANETS * 3)
        mine_mask = (res >= mine_cost) & active_60
        mask = torch.cat(
            [
                torch.ones(n, 1, dtype=torch.bool, device=self.device),       # proceed always valid
                (self.resources_mse >= astro_cost).unsqueeze(1),
                (self.resources_mse >= plasma_cost).unsqueeze(1),
                mine_mask,
            ],
            dim=1,
        )
        return obs, mask

    # --- reset ----------------------------------------------------------------------------------
    def _reset(self, tensordict=None):
        n = self.num_envs
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict.get("_reset").reshape(n)
        else:
            mask = torch.ones(n, dtype=torch.bool, device=self.device)

        m1 = mask.unsqueeze(1)
        z2d = torch.zeros_like(self.metal_lvl)
        self.metal_lvl = torch.where(m1, z2d, self.metal_lvl)
        self.crystal_lvl = torch.where(m1, z2d, self.crystal_lvl)
        self.deut_lvl = torch.where(m1, z2d, self.deut_lvl)
        self.astro_lvl = torch.where(mask, torch.zeros_like(self.astro_lvl), self.astro_lvl)
        self.plasma_lvl = torch.where(mask, torch.zeros_like(self.plasma_lvl), self.plasma_lvl)
        self.resources_mse = torch.where(mask, torch.zeros_like(self.resources_mse), self.resources_mse)
        self.points = torch.where(mask, torch.zeros_like(self.points), self.points)
        self.day = torch.where(mask, torch.zeros_like(self.day), self.day)
        self.step_ctr = torch.where(mask, torch.zeros_like(self.step_ctr), self.step_ctr)
        self.claimed = torch.where(mask.unsqueeze(1), torch.zeros_like(self.claimed), self.claimed)

        active = self._planet_active()
        production_mse, totM, totC, totD = self._production(active)
        obs, amask = self._observe(production_mse, totM, totC, totD, active)
        false = torch.zeros(n, 1, dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "observation": obs,
                "action_mask": amask,
                "done": false.clone(),
                "terminated": false.clone(),
                "truncated": false.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    # --- step -----------------------------------------------------------------------------------
    def _step(self, tensordict):
        n = self.num_envs
        a = tensordict.get("action").reshape(n).to(torch.int64)

        is_proceed = a == 0
        is_astro = a == 1
        is_plasma = a == 2
        is_mine = a >= 3
        planet_index = a // 3 - 1                       # -1 for actions 0/1/2, else 0..19
        pidx = planet_index.clamp(min=0, max=MAX_PLANETS - 1)
        res_type = a % 3

        # gather the targeted building's cost (MSE) and points-gained
        ml_sel = self.metal_lvl[self._ar, pidx]
        cl_sel = self.crystal_lvl[self._ar, pidx]
        dl_sel = self.deut_lvl[self._ar, pidx]
        ml_sel = ml_sel.clamp(max=self._lmax - 1)
        cl_sel = cl_sel.clamp(max=self._lmax - 1)
        dl_sel = dl_sel.clamp(max=self._lmax - 1)
        is_crystal = res_type == 1
        is_deut = res_type == 2
        mine_cost = torch.where(
            is_deut, self.luts.deut_cost_mse[dl_sel],
            torch.where(is_crystal, self.luts.crystal_cost_mse[cl_sel], self.luts.metal_cost_mse[ml_sel]),
        )
        mine_pts = torch.where(
            is_deut, self.luts.deut_cost_points[dl_sel],
            torch.where(is_crystal, self.luts.crystal_cost_points[cl_sel], self.luts.metal_cost_points[ml_sel]),
        )
        astro_lvl_c = self.astro_lvl.clamp(max=self._lmax - 1)
        plasma_lvl_c = self.plasma_lvl.clamp(max=self._lmax - 1)
        inf = torch.full_like(mine_cost, float("inf"))
        cost = torch.where(
            is_astro, self.luts.astro_cost_mse[astro_lvl_c],
            torch.where(is_plasma, self.luts.plasma_cost_mse[plasma_lvl_c],
                        torch.where(is_mine, mine_cost, inf)),
        )
        pts = torch.where(
            is_astro, self.luts.astro_cost_points[astro_lvl_c],
            torch.where(is_plasma, self.luts.plasma_cost_points[plasma_lvl_c],
                        torch.where(is_mine, mine_pts, torch.zeros_like(mine_pts))),
        )

        # locked planet (targeting a not-yet-unlocked planet) -> penalty, no upgrade
        planet_count = (((self.astro_lvl + 1) // 2) + 1).clamp(max=MAX_PLANETS)
        locked = is_mine & (planet_index > planet_count - 1)
        active_mine = is_mine & ~locked

        affordable = self.resources_mse >= cost
        do_upgrade = (is_astro | is_plasma | active_mine) & affordable

        # apply spend + points (branchless)
        spend = torch.where(do_upgrade, cost, torch.zeros_like(cost)).to(torch.int64)
        self.resources_mse = self.resources_mse - spend
        points_before = self.points
        gained = torch.where(do_upgrade, pts, torch.zeros_like(pts))
        self.points = self.points + gained

        # bump levels via masked one-hot scatter
        onehot = F.one_hot(pidx, MAX_PLANETS)  # (N,20) int64
        self.astro_lvl = self.astro_lvl + (do_upgrade & is_astro).to(torch.int64)
        self.plasma_lvl = self.plasma_lvl + (do_upgrade & is_plasma).to(torch.int64)
        self.metal_lvl = self.metal_lvl + onehot * (do_upgrade & active_mine & (res_type == 0)).unsqueeze(1)
        self.crystal_lvl = self.crystal_lvl + onehot * (do_upgrade & active_mine & is_crystal).unsqueeze(1)
        self.deut_lvl = self.deut_lvl + onehot * (do_upgrade & active_mine & is_deut).unsqueeze(1)

        # production tick (only on proceed)
        active = self._planet_active()
        production_mse, totM, totC, totD = self._production(active)
        self.resources_mse = self.resources_mse + torch.where(
            is_proceed, production_mse, torch.zeros_like(production_mse)
        )
        self.day = self.day + is_proceed.to(torch.int64)

        # exploration bucket bonus (per-env, claim once). Uses updated points.
        bucket = torch.floor(self.points / REWARD_DISTRIBUTION).to(torch.int64)
        bucket_c = bucket.clamp(min=0, max=N_BUCKETS - 1)
        already = self.claimed[self._ar, bucket_c]
        valid_claim = (bucket >= 0) & (bucket < N_BUCKETS) & do_upgrade & ~already
        explore = torch.where(
            valid_claim,
            (EXPLORATION_MAX_VALUE / N_BUCKETS) * bucket.to(torch.float64),
            torch.zeros_like(self.points),
        )
        if self.include_exploration:
            new_claim = already | valid_claim
            self.claimed = self.claimed.scatter(1, bucket_c.unsqueeze(1), new_claim.unsqueeze(1))

        # reward
        if self.reward_mode == "points":
            base = torch.log10(self.points + 1.0) - torch.log10(points_before + 1.0)
            reward = base + self.exploration_weight * explore
        else:  # "ogame" — exact C# reward
            upgrade_reward = torch.log10(gained + 1.0) + explore
            reward = torch.where(
                is_proceed,
                torch.full_like(self.points, 0.1),
                torch.where(do_upgrade, upgrade_reward, torch.full_like(self.points, -0.1)),
            )

        # build next observation + mask
        obs, amask = self._observe(production_mse, totM, totC, totD, active)

        self.step_ctr = self.step_ctr + 1
        truncated = (self.step_ctr > self.max_steps).unsqueeze(1)
        terminated = torch.zeros_like(truncated)
        done = truncated | terminated

        return TensorDict(
            {
                "observation": obs,
                "action_mask": amask,
                "reward": reward.reshape(n, 1).to(torch.float32),
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
