using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;
using OGameSim.Production;

namespace PlanningPlayer;

public static class Planner
{
    public static decimal Search(Player root, int horizon)
    {
        var clone = root.DeepClone();
        return Evaluate(clone, (int)clone.Day, horizon);
    }

    internal static decimal Evaluate(Player state, int day, int horizon)
    {
        if (day >= horizon)
            return state.Points;

        var actions = EnumerateActions(state);
        decimal best = decimal.MinValue;
        foreach (var action in actions)
        {
            var clone = state.DeepClone();
            Apply(clone, action);
            var value = Evaluate(clone, day + action.TimeCost, horizon);
            if (value > best)
                best = value;
        }
        return best;
    }

    internal static List<ActionCandidate> EnumerateActions(Player player)
    {
        var list = new List<ActionCandidate>();
        foreach (var planet in player.Planets)
        {
            if (player.Resources.CanSubtract(planet.MetalMine.UpgradeCost))
                list.Add(new ActionCandidate(planet.MetalMine, planet.MetalMine.UpgradeCost, planet.MetalMine.UpgradeIncreasePerDay, 1));
            if (player.Resources.CanSubtract(planet.CrystalMine.UpgradeCost))
                list.Add(new ActionCandidate(planet.CrystalMine, planet.CrystalMine.UpgradeCost, planet.CrystalMine.UpgradeIncreasePerDay, 1));
            if (player.Resources.CanSubtract(planet.DeuteriumSynthesizer.UpgradeCost))
                list.Add(new ActionCandidate(planet.DeuteriumSynthesizer, planet.DeuteriumSynthesizer.UpgradeCost, planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, 1));
        }

        var currentProduction = player.GetTodaysProduction();
        if (player.Resources.CanSubtract(player.PlasmaTechnology.UpgradeCost))
        {
            var plasmaClone = player.DeepClone();
            plasmaClone.PlasmaTechnology.Upgrade();
            var productionUpgrade = plasmaClone.GetTodaysProduction() - currentProduction;
            list.Add(new ActionCandidate(player.PlasmaTechnology, player.PlasmaTechnology.UpgradeCost, productionUpgrade, 1));
        }

        // Astrophysics upgrade (two levels to unlock a new planet).
        var astroCost = player.Astrophysics.UpgradeCost;
        var astroCopy = new Astrophysics();
        for (int j = 0; j < player.Astrophysics.Level + 1; j++)
        {
            astroCopy.Upgrade();
        }
        astroCost += astroCopy.UpgradeCost; // cost of second level

        var productionGain = player.Planets[0].MetalMine.TodaysProduction +
            player.Planets[0].CrystalMine.TodaysProduction +
            player.Planets[0].DeuteriumSynthesizer.TodaysProduction;

        if (player.Resources.CanSubtract(astroCost))
            list.Add(new ActionCandidate(player.Astrophysics, astroCost, productionGain, 1));

        list.Add(ActionCandidate.Wait());

        return list.OrderBy(a => CalculateRoi(a.Cost, a.Gain)).ToList();
    }

    internal static void Apply(Player player, ActionCandidate action)
    {
        if (action.Upgradable is null)
        {
            player.ProceedToNextDay();
            return;
        }

        if (!player.TrySpendResources(action.Cost))
            return; // action assumed affordable

        if (action.Upgradable == player.Astrophysics)
        {
            player.Astrophysics.Upgrade();
            player.Astrophysics.Upgrade();
        }
        else
        {
            action.Upgradable.Upgrade();
        }
    }

    internal static double CalculateRoi(Resources cost, Resources gain)
    {
        var weightedCost = (double)cost.ConvertToMetalValue();
        var weightedGain = (double)gain.ConvertToMetalValue();
        if (weightedGain == 0)
            return double.PositiveInfinity;
        return weightedCost / weightedGain;
    }
}

public readonly record struct ActionCandidate(IUpgradable? Upgradable, Resources Cost, Resources Gain, int TimeCost)
{
    public static ActionCandidate Wait() => new(null, new Resources(0,0,0), new Resources(0,0,0), 1);
}

