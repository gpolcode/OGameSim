using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OGameSim.Entities;
using OGameSim.Production;

namespace PlanningPlayer;

public static class Planner
{
    public static decimal Search(Player root, int horizon, int beamWidth = 5)
    {
        var clone = root.DeepClone();
        var cache = new ConcurrentDictionary<string, decimal>();
        return Evaluate(clone, (int)clone.Day, horizon, beamWidth, cache);
    }

    internal static decimal Evaluate(Player state, int day, int horizon, int beamWidth, ConcurrentDictionary<string, decimal> cache)
    {
        if (day >= horizon)
            return state.Points;

        var key = BuildKey(state, day);
        return cache.GetOrAdd(key, _ =>
        {
            var remaining = horizon - day;
            var actions = EnumerateActions(state, remaining);
            var subset = actions.Take(beamWidth).ToList();
            if (!subset.Any(a => a.Upgradable is null))
                subset.Add(ActionCandidate.Wait());

            var results = new ConcurrentBag<decimal>();
            Parallel.ForEach(subset, action =>
            {
                var clone = state.DeepClone();
                Apply(clone, action);
                var value = Evaluate(clone, day + action.TimeCost, horizon, beamWidth, cache);
                results.Add(value);
            });

            return results.Max();
        });
    }

    internal static List<ActionCandidate> EnumerateActions(Player player, int remainingDays)
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

        var ordered = list.OrderBy(a => CalculateRoi(a.Cost, a.Gain));
        return ordered
            .Where(a => CalculatePayback(a.Cost, a.Gain) <= remainingDays || a.Upgradable is null)
            .ToList();
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

    internal static double CalculatePayback(Resources cost, Resources gain)
    {
        var weightedCost = (double)cost.ConvertToMetalValue();
        var weightedGain = (double)gain.ConvertToMetalValue();
        if (weightedGain == 0) return double.PositiveInfinity;
        return weightedCost / weightedGain;
    }

    static string BuildKey(Player player, int day)
    {
        var sb = new StringBuilder();
        sb.Append(day).Append('|');
        var res = player.Resources;
        sb.Append(res.Metal).Append(',').Append(res.Crystal).Append(',').Append(res.Deuterium).Append('|');
        sb.Append(player.Astrophysics.Level).Append('|');
        sb.Append(player.PlasmaTechnology.Level).Append('|');
        foreach (var planet in player.Planets)
        {
            sb.Append(planet.MetalMine.Level).Append(',');
            sb.Append(planet.CrystalMine.Level).Append(',');
            sb.Append(planet.DeuteriumSynthesizer.Level).Append(';');
        }
        return sb.ToString();
    }
}

internal readonly record struct ActionCandidate(IUpgradable? Upgradable, Resources Cost, Resources Gain, int TimeCost)
{
    public static ActionCandidate Wait() => new(null, new Resources(0,0,0), new Resources(0,0,0), 1);
}

