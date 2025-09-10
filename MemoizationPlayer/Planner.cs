using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;
using OGameSim.Production;

namespace MemoizationPlayer;

public static class Planner
{
    public static decimal Search(Player root, int horizon)
    {
        var clone = root.DeepClone();
        var cache = new HashSet<string>();
        var stack = new Stack<(Player state, int day)>();
        stack.Push((clone, (int)clone.Day));
        decimal best = decimal.MinValue;
        var maxDay = (int)clone.Day;

        while (stack.Count > 0)
        {
            var (state, day) = stack.Pop();
            if (day > maxDay)
            {
                maxDay = day;
                Console.WriteLine($"Day {maxDay}/{horizon}");
                Console.Out.Flush();
            }
            if (day >= horizon)
            {
                if (state.Points > best) best = state.Points;
                continue;
            }

            var key = BuildKey(state, day);
            if (!cache.Add(key))
                continue;

            var actions = EnumerateActions(state);
            foreach (var action in actions)
            {
                var next = state.DeepClone();
                Apply(next, action);
                stack.Push((next, day + action.TimeCost));
            }
        }

        return best;
    }

    internal static string BuildKey(Player player, int day)
    {
        var res = player.Resources;
        var key = $"{day}|{res.Metal}|{res.Crystal}|{res.Deuterium}|";
        foreach (var planet in player.Planets)
        {
            key += $"{planet.MetalMine.Level}-{planet.CrystalMine.Level}-{planet.DeuteriumSynthesizer.Level};";
        }
        key += $"{player.PlasmaTechnology.Level}|{player.Astrophysics.Level}";
        return key;
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
