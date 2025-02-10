from pythonnet import load

load("coreclr", runtime_config="/home/elsahr/ogamesim/runtimeconfig.json")

import clr
clr.AddReference("/home/elsahr/ogamesim/Game.dll")

from OGameSim.Entities import *
from OGameSim.Production import *


def get_player_production(player, modifier):
    mine_production = Resources(0,0,0)
    for planet in player.Planets:
        mine_production += planet.MetalMine.TodaysProduction
        mine_production += planet.CrystalMine.TodaysProduction
        mine_production += planet.DeuteriumSynthesizer.TodaysProduction
    return mine_production + (mine_production * modifier)


def calculate_roi(cost, production_increase):
    weighted_cost = float(cost.ConvertToMetalValue())
    weighted_increase = float(production_increase.ConvertToMetalValue())
    return weighted_cost / weighted_increase


player = Player()

i = 0
while i < 8000:
    upgrade_and_increases = []

    for planet in player.Planets:
        upgrade_and_increases.append((planet.MetalMine, planet.MetalMine.UpgradeCost, planet.MetalMine.UpgradeIncreasePerDay))
        upgrade_and_increases.append((planet.CrystalMine, planet.CrystalMine.UpgradeCost, planet.CrystalMine.UpgradeIncreasePerDay))
        upgrade_and_increases.append((planet.DeuteriumSynthesizer, planet.DeuteriumSynthesizer.UpgradeCost, planet.DeuteriumSynthesizer.UpgradeIncreasePerDay))

    current_production = get_player_production(player, player.PlasmaTechnology.Modifier)
    upgraded_production = get_player_production(player, player.PlasmaTechnology.UpgradedModifier)
    production_upgrade = upgraded_production - current_production

    upgrade_and_increases.append((player.PlasmaTechnology, player.PlasmaTechnology.UpgradeCost, production_upgrade))

    production_upgrade = (
        player.Planets[0].MetalMine.TodaysProduction +
        player.Planets[0].CrystalMine.TodaysProduction +
        player.Planets[0].DeuteriumSynthesizer.TodaysProduction
    )

    astro_copy = Astrophysics()
    for _ in range(player.Astrophysics.Level):
        astro_copy.Upgrade()

    astro_cost = player.Astrophysics.UpgradeCost
    additional_steps_taken_for_astro = 1
    astro_copy.Upgrade()
    astro_cost += astro_copy.UpgradeCost

    metal_copy = MetalMine()
    for _ in range(player.Planets[0].MetalMine.Level):
        additional_steps_taken_for_astro += 1
        astro_cost += metal_copy.UpgradeCost
        metal_copy.Upgrade()

    crystal_mine_copy = CrystalMine()
    for _ in range(player.Planets[0].CrystalMine.Level):
        additional_steps_taken_for_astro += 1
        astro_cost += crystal_mine_copy.UpgradeCost
        crystal_mine_copy.Upgrade()

    deut_copy = DeuteriumSynthesizer(player.Planets[0].MaxTemperature)
    for _ in range(player.Planets[0].DeuteriumSynthesizer.Level):
        additional_steps_taken_for_astro += 1
        astro_cost += deut_copy.UpgradeCost
        deut_copy.Upgrade()

    upgrade_and_increases.append((player.Astrophysics, astro_cost, production_upgrade))

    upgrade_and_rois = [(u, c, calculate_roi(c, p)) for u, c, p in upgrade_and_increases]
    best_upgrade = min(upgrade_and_rois, key=lambda x: x[2])

    if player.TrySpendResources(best_upgrade[1]):
        if best_upgrade[0] == player.Astrophysics:
            player.Astrophysics.Upgrade()
            player.Astrophysics.Upgrade()

            planet_count = player.Planets.Count
            new_planet = player.Planets[planet_count - 1]
            for _ in range(player.Planets[0].MetalMine.Level):
                new_planet.MetalMine.Upgrade()
            for _ in range(player.Planets[0].CrystalMine.Level):
                new_planet.CrystalMine.Upgrade()
            for _ in range(player.Planets[0].DeuteriumSynthesizer.Level):
                new_planet.DeuteriumSynthesizer.Upgrade()

            i += 1
            i += additional_steps_taken_for_astro
        else:
            i += 1
            best_upgrade[0].Upgrade()
    else:
        i += 1
        player.ProceedToNextDay()

print(player.Points)
