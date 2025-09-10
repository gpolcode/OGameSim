using System.Collections.Generic;
using System.Reflection;
using OGameSim.Entities;
using OGameSim.Production;

namespace BranchBoundPlayer;

public static class PlayerExtensions
{
    public static Player DeepClone(this Player player)
    {
        var clone = new Player();
        SetProperty(clone, nameof(Player.Points), GetProperty<decimal>(player, nameof(Player.Points)));
        SetProperty(clone, nameof(Player.Day), GetProperty<uint>(player, nameof(Player.Day)));
        SetProperty(clone, nameof(Player.Resources), GetProperty<Resources>(player, nameof(Player.Resources)));

        CopyAstrophysics(player.Astrophysics, clone.Astrophysics);
        CopyPlasmaTechnology(player.PlasmaTechnology, clone.PlasmaTechnology);

        var planetsField = typeof(Player).GetField("_planets", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var originalPlanets = (List<Planet>)planetsField.GetValue(player)!;
        var clonedPlanets = new List<Planet>();
        foreach (var planet in originalPlanets)
        {
            clonedPlanets.Add(ClonePlanet(planet));
        }
        planetsField.SetValue(clone, clonedPlanets);

        var lastUpdatedField = typeof(Player).GetField("_lastUpdatedAstroLevel", BindingFlags.NonPublic | BindingFlags.Instance);
        lastUpdatedField?.SetValue(clone, lastUpdatedField.GetValue(player));

        return clone;
    }

    static Planet ClonePlanet(Planet original)
    {
        var planet = new Planet(original.MaxTemperature);
        CopyMine(original.MetalMine, planet.MetalMine);
        CopyMine(original.CrystalMine, planet.CrystalMine);
        CopyMine(original.DeuteriumSynthesizer, planet.DeuteriumSynthesizer);
        return planet;
    }

    static void CopyAstrophysics(Astrophysics source, Astrophysics target)
    {
        CopyProperty(source, target, nameof(Astrophysics.Level));
        CopyProperty(source, target, nameof(Astrophysics.UpgradeCost));
    }

    static void CopyPlasmaTechnology(PlasmaTechnology source, PlasmaTechnology target)
    {
        CopyProperty(source, target, nameof(PlasmaTechnology.Level));
        CopyProperty(source, target, nameof(PlasmaTechnology.Modifier));
        CopyProperty(source, target, nameof(PlasmaTechnology.UpgradedModifier));
        CopyProperty(source, target, nameof(PlasmaTechnology.UpgradeCost));
    }

    static void CopyMine(Mine source, Mine target)
    {
        CopyProperty(source, target, nameof(Mine.Level));
        CopyProperty(source, target, nameof(Mine.TodaysProduction));
        CopyProperty(source, target, nameof(Mine.UpgradeCost));
        CopyProperty(source, target, nameof(Mine.UpgradeIncreasePerDay));
    }

    static T GetProperty<T>(object obj, string name)
    {
        var prop = obj.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!;
        return (T)prop.GetValue(obj)!;
    }

    static void SetProperty(object obj, string name, object value)
    {
        var prop = obj.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!;
        var setter = prop.GetSetMethod(true)!;
        setter.Invoke(obj, new[] { value });
    }

    static void CopyProperty(object source, object target, string name)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy;
        var sourceProp = source.GetType().GetProperty(name, flags);
        var targetProp = target.GetType().GetProperty(name, flags);

        // ignore properties that cannot be copied (missing getter/setter)
        var getter = sourceProp?.GetGetMethod(true);
        var setter = targetProp?.GetSetMethod(true);
        if (getter is null || setter is null)
        {
            return;
        }

        var value = getter.Invoke(source, null);
        setter.Invoke(target, new[] { value });
    }
}

