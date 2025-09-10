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
        int maxDay = (int)clone.Day;
        return Evaluate(clone, maxDay, horizon, ref maxDay);
    }

    internal static decimal Evaluate(Player state, int day, int horizon, ref int maxDay)
    {
        if (day > maxDay)
        {
            maxDay = day;
            Console.WriteLine($"Day {maxDay}/{horizon}");
        }

        if (day >= horizon)
            return state.Points;

        var actions = EnumerateActions(state);
        decimal best = decimal.MinValue;
        foreach (var action in actions)
        {
            var clone = state.DeepClone();
            Apply(clone, action);
            var value = Evaluate(clone, day + action.TimeCost, horizon, ref maxDay);
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
            list.Add(new ActionCandidate(planet.MetalMine, planet.MetalMine.UpgradeCost, planet.MetalMine.UpgradeIncreasePerDay, 1));
            list.Add(new ActionCandidate(planet.CrystalMine, planet.CrystalMine.UpgradeCost, planet.CrystalMine.UpgradeIncreasePerDay, 1));
            list.Add(new ActionCandidate(planet.DeuteriumSynthesizer, planet.DeuteriumSynthesizer.UpgradeCost, planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, 1));
        }

        var currentProduction = player.GetTodaysProduction();
        var plasmaClone = player.DeepClone();
        plasmaClone.PlasmaTechnology.Upgrade();
        var productionUpgrade = plasmaClone.GetTodaysProduction() - currentProduction;
        list.Add(new ActionCandidate(player.PlasmaTechnology, player.PlasmaTechnology.UpgradeCost, productionUpgrade, 1));

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

        if (player.TrySpendResources(action.Cost))
        {
            action.Upgradable.Upgrade();
        }
        else
        {
            player.ProceedToNextDay();
        }
    }

    internal static double CalculateRoi(Resources cost, Resources gain)
    {
        var weightedCost = (double)cost.ConvertToMetalValue();
        var weightedGain = (double)gain.ConvertToMetalValue();
        return weightedCost / weightedGain;
    }
}

internal readonly record struct ActionCandidate(IUpgradable? Upgradable, Resources Cost, Resources Gain, int TimeCost)
{
    public static ActionCandidate Wait() => new(null, new Resources(0,0,0), new Resources(0,0,0), 1);
}

