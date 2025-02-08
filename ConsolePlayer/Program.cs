using System;
using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;
using OGameSim.Production;

var player = new Player();

for (int i = 0; i < 8000; i++)
{
    var upgradeAndIncreases =
        new List<(IUpgradable Upgradable, Resources Cost, Resources ProductionIncrease)>();

    foreach (var planet in player.Planets)
    {
        upgradeAndIncreases.Add(
            new(
                planet.MetalMine,
                planet.MetalMine.UpgradeCost,
                planet.MetalMine.UpgradeIncreasePerDay
            )
        );

        upgradeAndIncreases.Add(
            new(
                planet.CrystalMine,
                planet.CrystalMine.UpgradeCost,
                planet.CrystalMine.UpgradeIncreasePerDay
            )
        );

        upgradeAndIncreases.Add(
            new(
                planet.DeuteriumSynthesizer,
                planet.DeuteriumSynthesizer.UpgradeCost,
                planet.DeuteriumSynthesizer.UpgradeIncreasePerDay
            )
        );
    }

    var currentProduction = GetPlayerProduction(player, player.PlasmaTechnology.Modifier);
    var upgradedProduction = GetPlayerProduction(player, player.PlasmaTechnology.UpgradedModifier);
    var productionUpgrade = upgradedProduction - currentProduction;

    upgradeAndIncreases.Add(
        new(
            player.PlasmaTechnology,
            player.PlasmaTechnology.UpgradeCost,
            productionUpgrade
        )
    );

    productionUpgrade = player.Planets[0].MetalMine.TodaysProduction +
        player.Planets[0].CrystalMine.TodaysProduction +
        player.Planets[0].DeuteriumSynthesizer.TodaysProduction;

    var astroCopy = new Astrophysics();
    for (int j = 0; j < player.Astrophysics.Level; j++)
    {
        astroCopy.Upgrade();
    }

    var astroCost = player.Astrophysics.UpgradeCost;
    var additionalStepsTakenForAstro = 1;
    astroCopy.Upgrade();
    astroCost += astroCopy.UpgradeCost;

    var metalCopy = new MetalMine();
    for (int j = 0; j < player.Planets[0].MetalMine.Level; j++)
    {
        additionalStepsTakenForAstro++;
        astroCost += metalCopy.UpgradeCost;
        metalCopy.Upgrade();
    }

    var crystalMineCopy = new CrystalMine();
    for (int j = 0; j < player.Planets[0].CrystalMine.Level; j++)
    {
        additionalStepsTakenForAstro++;
        astroCost += crystalMineCopy.UpgradeCost;
        crystalMineCopy.Upgrade();
    }

    var deutCopy = new DeuteriumSynthesizer(player.Planets[0].MaxTemperature);
    for (int j = 0; j < player.Planets[0].DeuteriumSynthesizer.Level; j++)
    {
        additionalStepsTakenForAstro++;
        astroCost += deutCopy.UpgradeCost;
        deutCopy.Upgrade();
    }

    upgradeAndIncreases.Add(
        new(
            player.Astrophysics,
            astroCost,
            productionUpgrade
        )
    );

    var upgradeAndRois = new List<(IUpgradable Upgradable, Resources Cost, double Roi)>();
    foreach (var (upgradable, cost, productionIncrease) in upgradeAndIncreases)
    {
        upgradeAndRois.Add(new(upgradable, cost, CalculateRoi(cost, productionIncrease)));
    }

    var bestUpgrade = upgradeAndRois.MinBy(x => x.Roi);
    if (player.TrySpendResources(bestUpgrade.Cost))
    {
        if (bestUpgrade.Upgradable == player.Astrophysics)
        {
            player.Astrophysics.Upgrade();
            player.Astrophysics.Upgrade();

            var newPlanet = player.Planets.Last();
            for (int j = 0; j < player.Planets[0].MetalMine.Level; j++)
            {
                newPlanet.MetalMine.Upgrade();
            }

            for (int j = 0; j < player.Planets[0].CrystalMine.Level; j++)
            {
                newPlanet.CrystalMine.Upgrade();
            }

            for (int j = 0; j < player.Planets[0].DeuteriumSynthesizer.Level; j++)
            {
                newPlanet.DeuteriumSynthesizer.Upgrade();
            }

            // remove additional steps for the used upgrades for the mines
            i += additionalStepsTakenForAstro;
        }
        else
        {
            bestUpgrade.Upgradable.Upgrade();
        }
    }
    else
    {
        player.ProceedToNextDay();
    }
}

Console.WriteLine(player.Points);
Console.ReadLine();

static Resources GetPlayerProduction(Player player, ResourcesModifier modifier)
{
    Resources mineProduction = new();
    foreach (var planet in player.Planets)
    {
        mineProduction += planet.MetalMine.TodaysProduction;
        mineProduction += planet.CrystalMine.TodaysProduction;
        mineProduction += planet.DeuteriumSynthesizer.TodaysProduction;
    }

    return mineProduction + (mineProduction * modifier);
}

static double CalculateRoi(Resources cost, Resources productionIncrease)
{
    var weightedCost = (double)cost.ConvertToMetalValue();
    var weightedIncrease = (double)productionIncrease.ConvertToMetalValue();
    return weightedCost / weightedIncrease;
}
