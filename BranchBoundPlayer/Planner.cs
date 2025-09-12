using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;
using OGameSim.Production;

namespace BranchBoundPlayer;

public static class Planner
{
    public static decimal Search(Player root, int horizon)
    {
        var clone = root.DeepClone();
        var stack = new Stack<(Player state, int day)>();
        stack.Push((clone, (int)clone.Day));
        decimal best = decimal.MinValue;

        while (stack.Count > 0)
        {
            var (state, day) = stack.Pop();
            if (day >= horizon)
            {
                if (state.Points > best)
                {
                    best = state.Points;
                    Console.WriteLine($"New best score: {best}");
                }
                continue;
            }

            var actions = EnumerateActions(state);
            var maxPointsGain = actions.Max(a => a.PointsGain);
            var optimistic = state.Points + (horizon - day) * maxPointsGain;
            if (optimistic <= best)
                continue;

            foreach (var action in actions)
            {
                var next = state.DeepClone();
                Apply(next, action);
                stack.Push((next, day + action.TimeCost));
            }
        }

        return best;
    }

    internal static decimal Evaluate(Player state, int day, int horizon, ref decimal best)
    {
        if (day >= horizon)
        {
            if (state.Points > best)
            {
                best = state.Points;
                Console.WriteLine($"New best score: {best}");
            }
            return state.Points;
        }

        var actions = EnumerateActions(state);
        var maxPointsGain = actions.Max(a => a.PointsGain);
        var optimistic = state.Points + (horizon - day) * maxPointsGain;
        if (optimistic <= best)
            return decimal.MinValue;

        decimal bestLocal = decimal.MinValue;
        foreach (var action in actions)
        {
            var clone = state.DeepClone();
            Apply(clone, action);
            var value = Evaluate(clone, day + action.TimeCost, horizon, ref best);
            if (value > bestLocal)
                bestLocal = value;
        }
        if (bestLocal > best)
        {
            best = bestLocal;
            Console.WriteLine($"New best score: {best}");
        }
        return bestLocal;
    }

    internal static List<ActionCandidate> EnumerateActions(Player player)
    {
        var list = new List<ActionCandidate>();
        foreach (var planet in player.Planets)
        {
            list.Add(new ActionCandidate(planet.MetalMine, planet.MetalMine.UpgradeCost, planet.MetalMine.UpgradeIncreasePerDay, 1, CalculatePointsGain(planet.MetalMine.UpgradeCost)));
            list.Add(new ActionCandidate(planet.CrystalMine, planet.CrystalMine.UpgradeCost, planet.CrystalMine.UpgradeIncreasePerDay, 1, CalculatePointsGain(planet.CrystalMine.UpgradeCost)));
            list.Add(new ActionCandidate(planet.DeuteriumSynthesizer, planet.DeuteriumSynthesizer.UpgradeCost, planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, 1, CalculatePointsGain(planet.DeuteriumSynthesizer.UpgradeCost)));
        }

        var currentProduction = player.GetTodaysProduction();
        var plasmaClone = player.DeepClone();
        plasmaClone.PlasmaTechnology.Upgrade();
        var productionUpgrade = plasmaClone.GetTodaysProduction() - currentProduction;
        list.Add(new ActionCandidate(player.PlasmaTechnology, player.PlasmaTechnology.UpgradeCost, productionUpgrade, 1, CalculatePointsGain(player.PlasmaTechnology.UpgradeCost)));

        list.Add(ActionCandidate.Wait());

        return list.OrderBy(a => CalculateRoi(a.Cost, a.Gain)).ToList();
    }

    static decimal CalculatePointsGain(Resources cost)
    {
        return cost.Metal / 1000m + cost.Crystal / 1000m + cost.Deuterium / 1000m;
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

internal readonly record struct ActionCandidate(IUpgradable? Upgradable, Resources Cost, Resources Gain, int TimeCost, decimal PointsGain)
{
    public static ActionCandidate Wait() => new(null, new Resources(0,0,0), new Resources(0,0,0), 1, 0);
}
