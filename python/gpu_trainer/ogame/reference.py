"""Scalar pure-Python port of the C# ``OGameSim.Core`` economy — the correctness oracle.

This is a faithful 1:1 translation of ``csharp/OGameSim.Core/`` so the batched tensor env
(``ogame/env.py``) and the lookup tables (``ogame/luts.py``) can be checked against it. It deliberately
mirrors the C# class structure (``Resources``, ``Mine`` subclasses, ``Player``, ``Foo``) and its exact
rounding so it is easy to diff against the source. No torch — plain Python ints / ``Decimal`` / floats,
which reproduce C# ``ulong`` / ``decimal`` / ``double`` arithmetic for the ranges we exercise.

References (file:line) point at the C# original.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Tuple

# Planet temperature used by the game: every planet is `new Planet(-115)` (Player.cs:20).
GAME_PLANET_MAX_TEMPERATURE = -115

# Exploration reward buckets (Foo.cs:76-101).
REWARD_DISTRIBUTION = 5_000_000
EXPLORATION_BUCKET_COUNT = 300_000_000 // REWARD_DISTRIBUTION  # 60
EXPLORATION_MAX_VALUE = 25.0


# --------------------------------------------------------------------------------------------------
# Resources / ResourceWeight / ResourcesModifier  (Production/*.cs)
# --------------------------------------------------------------------------------------------------
# ResourceWeight.cs: metal=1, crystal=2, deuterium=3.
METAL_VALUE = 1
CRYSTAL_VALUE = 2
DEUTERIUM_VALUE = 3


@dataclass(frozen=True)
class Resources:
    """readonly record struct Resources(ulong Metal, ulong Crystal, ulong Deuterium) — Resources.cs."""

    metal: int = 0
    crystal: int = 0
    deuterium: int = 0

    def __add__(self, other: "Resources") -> "Resources":
        return Resources(
            self.metal + other.metal,
            self.crystal + other.crystal,
            self.deuterium + other.deuterium,
        )

    def __sub__(self, other: "Resources") -> "Resources":
        return Resources(
            self.metal - other.metal,
            self.crystal - other.crystal,
            self.deuterium - other.deuterium,
        )

    def apply_modifier(self, mod: "ResourcesModifier") -> "Resources":
        """Resources * ResourcesModifier — floors each component (Resources.cs:13-18)."""
        return Resources(
            math.floor(self.metal * mod.metal),
            math.floor(self.crystal * mod.crystal),
            math.floor(self.deuterium * mod.deuterium),
        )

    def can_subtract(self, other: "Resources") -> bool:
        """CanSubtract — compares metal-equivalent values (Resources.cs:20-23)."""
        return self.convert_to_metal_value() >= other.convert_to_metal_value()

    def convert_to_metal_value(self) -> int:
        """ConvertToMetalValue (MSE) — metal*1 + crystal*2 + deut*3 (Resources.cs:25-32)."""
        return (
            self.metal * METAL_VALUE
            + self.crystal * CRYSTAL_VALUE
            + self.deuterium * DEUTERIUM_VALUE
        )


@dataclass(frozen=True)
class ResourcesModifier:
    """readonly record struct ResourcesModifier(decimal Metal, Crystal, Deuterium) — ResourcesModifier.cs.

    Stored as exact ``Decimal`` to mirror C# ``decimal`` (the plasma modifier is a small rational).
    """

    metal: Decimal = Decimal(0)
    crystal: Decimal = Decimal(0)
    deuterium: Decimal = Decimal(0)

    def __sub__(self, other: "ResourcesModifier") -> "ResourcesModifier":
        return ResourcesModifier(
            self.metal - other.metal,
            self.crystal - other.crystal,
            self.deuterium - other.deuterium,
        )


# --------------------------------------------------------------------------------------------------
# IUpgradable / Mine and subclasses  (Entities/*.cs)
# --------------------------------------------------------------------------------------------------
class _Mine:
    """abstract class Mine : IUpgradable — Mine.cs.

    Replicates the *mutable accumulation* of TodaysProduction so any rounding behaviour matches the
    C# exactly (rather than relying on the closed form).
    """

    def __init__(self, base_production: Resources):
        self.level = 0
        self.todays_production = base_production
        self.upgrade_cost = self._calculate_upgrade_cost()
        self.upgrade_increase_per_day = self._calculate_upgraded_production() - self.todays_production

    def upgrade(self) -> None:
        # Mine.cs:14-20 — note: increment is the PRE-stored value, then cost/increment recomputed.
        self.level += 1
        self.todays_production = self.todays_production + self.upgrade_increase_per_day
        self.upgrade_cost = self._calculate_upgrade_cost()
        self.upgrade_increase_per_day = self._calculate_upgraded_production() - self.todays_production

    def _calculate_upgrade_cost(self) -> Resources:  # pragma: no cover - abstract
        raise NotImplementedError

    def _calculate_upgraded_production(self) -> Resources:  # pragma: no cover - abstract
        raise NotImplementedError


class MetalMine(_Mine):
    """MetalMine.cs — base production 30*24 metal/day."""

    def __init__(self):
        super().__init__(Resources(30 * 24, 0, 0))

    def _calculate_upgrade_cost(self) -> Resources:
        metal_cost = 60 * math.pow(1.5, self.level)
        crystal_cost = 15 * math.pow(1.5, self.level)
        return Resources(math.floor(metal_cost), math.floor(crystal_cost), 0)

    def _calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        production_per_hour = 30 * next_level * math.pow(1.1, next_level)
        production_per_day = round(production_per_hour) * 24  # Math.Round = banker's, like Python round
        return Resources(production_per_day, 0, 0)


class CrystalMine(_Mine):
    """CrystalMine.cs — base production 15*24 crystal/day."""

    def __init__(self):
        super().__init__(Resources(0, 15 * 24, 0))

    def _calculate_upgrade_cost(self) -> Resources:
        metal_cost = 48 * math.pow(1.6, self.level)
        crystal_cost = 24 * math.pow(1.6, self.level)
        return Resources(math.ceil(metal_cost), math.ceil(crystal_cost), 0)

    def _calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        production_per_hour = 20 * next_level * math.pow(1.1, next_level)
        production_per_day = math.floor(production_per_hour) * 24
        return Resources(0, production_per_day, 0)


class DeuteriumSynthesizer(_Mine):
    """DeuteriumSynthesizer.cs — temperature-dependent; base production 0."""

    def __init__(self, planet_max_temperature: int):
        self._planet_average_temperature = planet_max_temperature - 20
        super().__init__(Resources())

    def _calculate_upgrade_cost(self) -> Resources:
        metal_cost = 225 * math.pow(1.5, self.level)
        crystal_cost = 75 * math.pow(1.5, self.level)
        return Resources(round(metal_cost), round(crystal_cost), 0)

    def _calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        production_per_hour = (
            20
            * next_level
            * math.pow(1.1, next_level)
            * (0.68 - 0.002 * self._planet_average_temperature)
        )
        production_per_day = math.floor(production_per_hour) * 24
        return Resources(0, 0, production_per_day)


class Astrophysics:
    """Astrophysics.cs — cost ratio metal:crystal:deut = common:2x:common."""

    def __init__(self):
        self.level = 0
        self.upgrade_cost = self._cost()

    def upgrade(self) -> None:
        self.level += 1
        self.upgrade_cost = self._cost()

    def _cost(self) -> Resources:
        common = math.floor(4000 * math.pow(1.75, self.level))
        crystal = math.floor(8000 * math.pow(1.75, self.level))
        return Resources(common, crystal, common)


class PlasmaTechnology:
    """PlasmaTechnology.cs — production modifier + geometric cost."""

    def __init__(self):
        self.level = 0
        self.upgrade_cost = self._cost()
        self.modifier = self._modifier(self.level)
        self.upgraded_modifier = self._modifier(self.level + 1)

    def upgrade(self) -> None:
        self.level += 1
        self.upgrade_cost = self._cost()
        self.modifier = self._modifier(self.level)
        self.upgraded_modifier = self._modifier(self.level + 1)

    @staticmethod
    def _modifier(level: int) -> ResourcesModifier:
        # level * 1m/100, level * 0.66m/100, level * 0.33m/100 (exact decimals).
        return ResourcesModifier(
            Decimal(level) * Decimal(1) / Decimal(100),
            Decimal(level) * Decimal("0.66") / Decimal(100),
            Decimal(level) * Decimal("0.33") / Decimal(100),
        )

    def _cost(self) -> Resources:
        pow2 = 2 ** self.level  # (ulong)Math.Pow(2, Level) — exact for the relevant range
        return Resources(2000 * pow2, 4000 * pow2, 1000 * pow2)


class Planet:
    """Planet.cs — three mines, fixed max temperature."""

    def __init__(self, max_temperature: int):
        self.max_temperature = max_temperature
        self.metal_mine = MetalMine()
        self.crystal_mine = CrystalMine()
        self.deuterium_synthesizer = DeuteriumSynthesizer(max_temperature)


# --------------------------------------------------------------------------------------------------
# Player  (Entities/Player.cs)
# --------------------------------------------------------------------------------------------------
class Player:
    def __init__(self):
        self.astrophysics = Astrophysics()
        self.plasma_technology = PlasmaTechnology()
        self.points = Decimal(0)
        self.day = 0
        self.resources = Resources()
        self._planets: List[Planet] = []
        self._last_updated_astro_level = None

    @property
    def planets(self) -> List[Planet]:
        """Lazily grow the planet list: ceil(astroLevel/2)+1 (Player.cs:11-28)."""
        if self.astrophysics.level != self._last_updated_astro_level:
            target = math.ceil(self.astrophysics.level / 2) + 1
            while len(self._planets) < target:
                self._planets.append(Planet(GAME_PLANET_MAX_TEMPERATURE))
            self._last_updated_astro_level = self.astrophysics.level
        return self._planets

    def add_resources(self, resources: Resources) -> None:
        self.resources = self.resources + resources

    def try_spend_resources(self, cost: Resources) -> bool:
        """Player.cs:42-58 — affordability is MSE; on spend, resources collapse to (MSE, 0, 0)."""
        if not self.resources.can_subtract(cost):
            return False

        resources_value = self.resources.convert_to_metal_value()
        cost_value = cost.convert_to_metal_value()
        self.resources = Resources(resources_value - cost_value, 0, 0)

        self.points += Decimal(cost.metal) / Decimal(1000)
        self.points += Decimal(cost.crystal) / Decimal(1000)
        self.points += Decimal(cost.deuterium) / Decimal(1000)
        return True

    def proceed_to_next_day(self) -> None:
        self.day += 1
        self.add_resources(self.get_todays_production())

    def get_todays_production(self) -> Resources:
        """Player.cs:67-85 — sum mine production over planets, then add plasma-modified production."""
        mine_production = Resources()
        for planet in self.planets:
            mine_production = mine_production + planet.metal_mine.todays_production
            mine_production = mine_production + planet.crystal_mine.todays_production
            mine_production = mine_production + planet.deuterium_synthesizer.todays_production

        modifier_production = mine_production.apply_modifier(self.plasma_technology.modifier)
        return mine_production + modifier_production


# --------------------------------------------------------------------------------------------------
# Foo  (Production/Foo.cs) — exploration reward, ApplyAction, UpdateState
# --------------------------------------------------------------------------------------------------
@dataclass
class _ExplorationRewards:
    """Per-player port of Foo's static reward dictionary (Foo.cs:24-101).

    The C# dict is process-global; the faithful single-player semantics are per-player buckets, which
    is also how the batched env models it (per-env). Buckets beyond 300M do not exist — C# would
    KeyError; we simply grant nothing (never capping points).
    """

    redeemed: List[bool] = field(default_factory=lambda: [False] * EXPLORATION_BUCKET_COUNT)

    def claim(self, points: Decimal) -> float:
        bucket_index = int(points // REWARD_DISTRIBUTION)
        if bucket_index < 0 or bucket_index >= EXPLORATION_BUCKET_COUNT:
            return 0.0
        if self.redeemed[bucket_index]:
            return 0.0
        self.redeemed[bucket_index] = True
        return EXPLORATION_MAX_VALUE / EXPLORATION_BUCKET_COUNT * bucket_index


N_ACTIONS = 63
OBS_DIM = 125
MAX_PLANETS = 20  # observation has 20 planet slots (5 + 20*6 = 125)


def exploration_value(bucket_index: int) -> float:
    """The reward value for a given bucket index (Foo.cs:81-91)."""
    return EXPLORATION_MAX_VALUE / EXPLORATION_BUCKET_COUNT * bucket_index


def apply_action(player: Player, rewards: _ExplorationRewards, action: int) -> Tuple[float, bool]:
    """Foo.ApplyAction (Foo.cs:103-157). Returns (reward, terminated)."""

    def penalty() -> Tuple[float, bool]:
        return (-0.1, False)

    def try_upgrade(upgradable) -> Tuple[float, bool]:
        current_points = player.points
        if player.try_spend_resources(upgradable.upgrade_cost):
            upgradable.upgrade()
            gained_points = float(player.points - current_points)
            upgrade_reward = math.log10(gained_points + 1)
            exploration_reward = rewards.claim(player.points)
            return (upgrade_reward + exploration_reward, False)
        return penalty()

    def proceed_to_next_day() -> Tuple[float, bool]:
        player.proceed_to_next_day()
        return (0.1, False)

    planet_index = math.floor(action / 3) - 1
    if planet_index > len(player.planets) - 1:
        return penalty()

    mod = action % 3
    if action == 0:
        return proceed_to_next_day()
    if action == 1:
        return try_upgrade(player.astrophysics)
    if action == 2:
        return try_upgrade(player.plasma_technology)
    if mod == 0:
        return try_upgrade(player.planets[planet_index].metal_mine)
    if mod == 1:
        return try_upgrade(player.planets[planet_index].crystal_mine)
    return try_upgrade(player.planets[planet_index].deuterium_synthesizer)


def update_state(player: Player) -> List[float]:
    """Foo.UpdateState (Foo.cs:159-206) — returns the 125-D observation (metal-equivalent doubles).

    Inactive planet slots stay 0 (the C# buffer is zero-initialised and only active planets written).
    """
    state = [0.0] * OBS_DIM
    todays_production = player.get_todays_production()

    state[0] = float(player.resources.convert_to_metal_value())
    state[1] = float(todays_production.convert_to_metal_value())
    state[2] = float(player.astrophysics.upgrade_cost.convert_to_metal_value())
    state[3] = float(player.plasma_technology.upgrade_cost.convert_to_metal_value())

    plasma_delta_mod = player.plasma_technology.upgraded_modifier - player.plasma_technology.modifier
    state[4] = float(todays_production.apply_modifier(plasma_delta_mod).convert_to_metal_value())

    idx = 5
    for planet in player.planets:
        if idx + 6 > OBS_DIM:
            break  # never exceeds 20 planets in practice
        state[idx] = float(planet.metal_mine.upgrade_cost.convert_to_metal_value())
        state[idx + 1] = float(planet.metal_mine.upgrade_increase_per_day.convert_to_metal_value())
        state[idx + 2] = float(planet.crystal_mine.upgrade_cost.convert_to_metal_value())
        state[idx + 3] = float(planet.crystal_mine.upgrade_increase_per_day.convert_to_metal_value())
        state[idx + 4] = float(planet.deuterium_synthesizer.upgrade_cost.convert_to_metal_value())
        state[idx + 5] = float(
            planet.deuterium_synthesizer.upgrade_increase_per_day.convert_to_metal_value()
        )
        idx += 6

    return state


def action_mask(player: Player) -> List[bool]:
    """Valid-action mask used by the aligned (points) reward's action masking.

    proceed always valid; astro/plasma valid iff affordable; mine[p,res] valid iff the planet is
    unlocked AND affordable. Mirrors the affordability / planet-count logic of ApplyAction so the
    batched env's mask can be diff-tested against it.
    """
    mask = [False] * N_ACTIONS
    mask[0] = True  # proceed always available
    mask[1] = player.resources.can_subtract(player.astrophysics.upgrade_cost)
    mask[2] = player.resources.can_subtract(player.plasma_technology.upgrade_cost)

    planets = player.planets
    for action in range(3, N_ACTIONS):
        planet_index = action // 3 - 1
        if planet_index > len(planets) - 1:
            continue
        planet = planets[planet_index]
        building = (planet.metal_mine, planet.crystal_mine, planet.deuterium_synthesizer)[action % 3]
        mask[action] = player.resources.can_subtract(building.upgrade_cost)
    return mask


class ReferenceEnv:
    """Convenience single-episode driver mirroring grid_world.py (max 8000 action-steps)."""

    MAX_STEPS = 8000

    def __init__(self):
        self.reset()

    def reset(self) -> List[float]:
        self.player = Player()
        self.rewards = _ExplorationRewards()
        self.step_counter = 0
        return update_state(self.player)

    def step(self, action: int):
        reward, terminated = apply_action(self.player, self.rewards, int(action))
        obs = update_state(self.player)
        self.step_counter += 1
        terminated = terminated or self.step_counter > self.MAX_STEPS
        if terminated:
            reward = 0.0
        return obs, reward, terminated, update_state(self.player)
