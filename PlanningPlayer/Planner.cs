using System.Collections.Generic;
using System.Linq;
using System.Text;
using OGameSim.Entities;
using OGameSim.Production;

namespace PlanningPlayer;

public static class Planner
{
    public static decimal Search(Player root, int horizon, int bucketSize = 1000, int bucketCount = 50)
    {
        var start = root.DeepClone();
        var current = new Dictionary<string, Player>
        {
            [BuildKey(start, bucketSize, bucketCount)] = start
        };

        for (var day = (int)start.Day; day < horizon; day++)
        {
            var next = new Dictionary<string, Player>();
            foreach (var state in current.Values)
            {
                var remaining = horizon - day;
                var actions = EnumerateActions(state, remaining);
                foreach (var action in actions)
                {
                    var clone = state.DeepClone();
                    Apply(clone, action);
                    var key = BuildKey(clone, bucketSize, bucketCount);
                    if (!next.TryGetValue(key, out var existing) || clone.Points > existing.Points)
                        next[key] = clone;
                }
            }
            current = next;
        }

        return current.Values.Max(p => p.Points);
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
            action.Upgradable.Upgrade();
        else
            player.ProceedToNextDay();
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

    internal static string BuildKey(Player player, int bucketSize, int bucketCount)
    {
        int Bucket(decimal value)
        {
            var index = (int)(value / bucketSize);
            if (index >= bucketCount) index = bucketCount - 1;
            return index;
        }

        var sb = new StringBuilder();
        var res = player.Resources;
        sb.Append(Bucket(res.Metal)).Append(',');
        sb.Append(Bucket(res.Crystal)).Append(',');
        sb.Append(Bucket(res.Deuterium)).Append('|');
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
