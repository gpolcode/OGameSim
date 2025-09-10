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
        for (int i = 0; i < player.Planets.Count; i++)
        {
            var planet = player.Planets[i];
            if (player.Resources.CanSubtract(planet.MetalMine.UpgradeCost))
                list.Add(new ActionCandidate(ActionType.MetalMine, i, planet.MetalMine.UpgradeCost, planet.MetalMine.UpgradeIncreasePerDay, 1));
            if (player.Resources.CanSubtract(planet.CrystalMine.UpgradeCost))
                list.Add(new ActionCandidate(ActionType.CrystalMine, i, planet.CrystalMine.UpgradeCost, planet.CrystalMine.UpgradeIncreasePerDay, 1));
            if (player.Resources.CanSubtract(planet.DeuteriumSynthesizer.UpgradeCost))
                list.Add(new ActionCandidate(ActionType.DeuteriumSynthesizer, i, planet.DeuteriumSynthesizer.UpgradeCost, planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, 1));
        }

        var currentProduction = player.GetTodaysProduction();
        if (player.Resources.CanSubtract(player.PlasmaTechnology.UpgradeCost))
        {
            var plasmaClone = player.DeepClone();
            plasmaClone.PlasmaTechnology.Upgrade();
            var productionUpgrade = plasmaClone.GetTodaysProduction() - currentProduction;
            list.Add(new ActionCandidate(ActionType.PlasmaTechnology, -1, player.PlasmaTechnology.UpgradeCost, productionUpgrade, 1));
        }

        if (player.Resources.CanSubtract(player.Astrophysics.UpgradeCost))
        {
            var astroClone = player.DeepClone();
            astroClone.Astrophysics.Upgrade();
            var productionUpgrade = astroClone.GetTodaysProduction() - currentProduction;
            list.Add(new ActionCandidate(ActionType.Astrophysics, -1, player.Astrophysics.UpgradeCost, productionUpgrade, 1));
        }

        list.Add(ActionCandidate.Wait());

        return list.OrderBy(a => CalculateRoi(a.Cost, a.Gain)).ToList();
    }

    internal static void Apply(Player player, ActionCandidate action)
    {
        switch (action.Type)
        {
            case ActionType.Wait:
                player.ProceedToNextDay();
                break;

            case ActionType.MetalMine:
                if (player.TrySpendResources(action.Cost))
                {
                    player.Planets[action.PlanetIndex].MetalMine.Upgrade();
                }
                break;

            case ActionType.CrystalMine:
                if (player.TrySpendResources(action.Cost))
                {
                    player.Planets[action.PlanetIndex].CrystalMine.Upgrade();
                }
                break;

            case ActionType.DeuteriumSynthesizer:
                if (player.TrySpendResources(action.Cost))
                {
                    player.Planets[action.PlanetIndex].DeuteriumSynthesizer.Upgrade();
                }
                break;

            case ActionType.PlasmaTechnology:
                if (player.TrySpendResources(action.Cost))
                {
                    player.PlasmaTechnology.Upgrade();
                }
                break;

            case ActionType.Astrophysics:
                if (player.TrySpendResources(action.Cost))
                {
                    player.Astrophysics.Upgrade();
                }
                break;
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

public enum ActionType
{
    Wait,
    MetalMine,
    CrystalMine,
    DeuteriumSynthesizer,
    PlasmaTechnology,
    Astrophysics
}

public readonly record struct ActionCandidate(ActionType Type, int PlanetIndex, Resources Cost, Resources Gain, int TimeCost)
{
    public static ActionCandidate Wait() => new(ActionType.Wait, -1, new Resources(0,0,0), new Resources(0,0,0), 1);
}

