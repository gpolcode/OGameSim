"""Integration test: the fixed ROI strategy from ``csharp/OGameSim.Console/Program.cs``.

That console driver runs a deterministic "best return-on-investment" agent for 8000 steps and the C#
build prints a precalculated final score of **266316720.384** points. This test ports the exact same
strategy onto the Python economy (``ogame/reference.py``) and asserts it lands on that scalar — an
end-to-end check of the economy + points accounting against a known C# value, with no .NET involved.

(The strategy uses a compound astrophysics move — buy two astro levels and clone planet[0]'s mines onto
the freshly unlocked planet, all under one bundled cost — which is not a single ``Foo.ApplyAction``
action, so it is expressed directly on the economy. The batched env shares that economy and is proven
equal to the reference by ``test_parity.py``.)
"""

from decimal import Decimal

from ogame import reference as ref

EXPECTED_POINTS = Decimal("266316720.384")
N_STEPS = 8000


def _player_production(player, modifier):
    """GetPlayerProduction(player, modifier) — Program.cs:147-158."""
    mine = ref.Resources()
    for planet in player.planets:
        mine = mine + planet.metal_mine.todays_production
        mine = mine + planet.crystal_mine.todays_production
        mine = mine + planet.deuterium_synthesizer.todays_production
    return mine + mine.apply_modifier(modifier)


def _roi(cost, increase):
    """CalculateRoi(cost, increase) = cost_MSE / increase_MSE (Program.cs:160-165).

    C# double division by zero yields +inf (never NaN here, since every upgrade cost MSE > 0).
    """
    weighted_cost = float(cost.convert_to_metal_value())
    weighted_increase = float(increase.convert_to_metal_value())
    return weighted_cost / weighted_increase if weighted_increase != 0 else float("inf")


def run_console_strategy():
    """Faithful port of Program.cs — returns the final ``player.Points`` (Decimal)."""
    player = ref.Player()
    i = 0
    while i < N_STEPS:
        candidates = []  # (tag, upgradable, cost, increase)
        for planet in player.planets:
            candidates.append(("mine", planet.metal_mine,
                               planet.metal_mine.upgrade_cost, planet.metal_mine.upgrade_increase_per_day))
            candidates.append(("mine", planet.crystal_mine,
                               planet.crystal_mine.upgrade_cost, planet.crystal_mine.upgrade_increase_per_day))
            candidates.append(("mine", planet.deuterium_synthesizer,
                               planet.deuterium_synthesizer.upgrade_cost,
                               planet.deuterium_synthesizer.upgrade_increase_per_day))

        # plasma: increase = production delta between current and upgraded modifier
        current = _player_production(player, player.plasma_technology.modifier)
        upgraded = _player_production(player, player.plasma_technology.upgraded_modifier)
        candidates.append(("plasma", player.plasma_technology,
                           player.plasma_technology.upgrade_cost, upgraded - current))

        # astrophysics: compound cost (two astro levels + cloning planet[0]'s mines onto a new planet),
        # with "increase" = planet[0]'s total production. (Program.cs:53-98)
        p0 = player.planets[0]
        astro_increase = (
            p0.metal_mine.todays_production
            + p0.crystal_mine.todays_production
            + p0.deuterium_synthesizer.todays_production
        )
        astro_copy = ref.Astrophysics()
        for _ in range(player.astrophysics.level):
            astro_copy.upgrade()
        astro_cost = player.astrophysics.upgrade_cost
        extra_steps = 1
        astro_copy.upgrade()
        astro_cost = astro_cost + astro_copy.upgrade_cost
        metal_copy = ref.MetalMine()
        for _ in range(p0.metal_mine.level):
            extra_steps += 1
            astro_cost = astro_cost + metal_copy.upgrade_cost
            metal_copy.upgrade()
        crystal_copy = ref.CrystalMine()
        for _ in range(p0.crystal_mine.level):
            extra_steps += 1
            astro_cost = astro_cost + crystal_copy.upgrade_cost
            crystal_copy.upgrade()
        deut_copy = ref.DeuteriumSynthesizer(p0.max_temperature)
        for _ in range(p0.deuterium_synthesizer.level):
            extra_steps += 1
            astro_cost = astro_cost + deut_copy.upgrade_cost
            deut_copy.upgrade()
        candidates.append(("astro", player.astrophysics, astro_cost, astro_increase))

        # MinBy(Roi) — lowest cost-per-production; Python min returns the first minimum, like C# MinBy.
        tag, upgradable, cost, _increase = min(candidates, key=lambda c: _roi(c[2], c[3]))

        if player.try_spend_resources(cost):
            if tag == "astro":
                player.astrophysics.upgrade()
                player.astrophysics.upgrade()
                new_planet = player.planets[-1]
                for _ in range(p0.metal_mine.level):
                    new_planet.metal_mine.upgrade()
                for _ in range(p0.crystal_mine.level):
                    new_planet.crystal_mine.upgrade()
                for _ in range(p0.deuterium_synthesizer.level):
                    new_planet.deuterium_synthesizer.upgrade()
                i += extra_steps  # account for the bundled upgrades (Program.cs:131)
            else:
                upgradable.upgrade()
        else:
            player.proceed_to_next_day()
        i += 1

    return player.points


def test_console_strategy_matches_csharp_precalculated_points():
    assert run_console_strategy() == EXPECTED_POINTS
