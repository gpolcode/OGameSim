using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;
using OGameSim.Entities;
using OGameSim.Production;

namespace PlanningPlayer;

public static class PlayerExtensions
{
    static readonly FieldInfo PlanetsField = typeof(Player).GetField("_planets", BindingFlags.NonPublic | BindingFlags.Instance)!;
    static readonly FieldInfo? LastUpdatedField = typeof(Player).GetField("_lastUpdatedAstroLevel", BindingFlags.NonPublic | BindingFlags.Instance);

    public static Player DeepClone(this Player player)
    {
        var clone = new Player();
        SetProperty(clone, nameof(Player.Points), GetProperty<decimal>(player, nameof(Player.Points)));
        SetProperty(clone, nameof(Player.Day), GetProperty<uint>(player, nameof(Player.Day)));
        SetProperty(clone, nameof(Player.Resources), GetProperty<Resources>(player, nameof(Player.Resources)));

        CopyAstrophysics(player.Astrophysics, clone.Astrophysics);
        CopyPlasmaTechnology(player.PlasmaTechnology, clone.PlasmaTechnology);

        var originalPlanets = (List<Planet>)PlanetsField.GetValue(player)!;
        var clonedPlanets = new List<Planet>(originalPlanets.Count);
        foreach (var planet in originalPlanets)
        {
            clonedPlanets.Add(ClonePlanet(planet));
        }
        PlanetsField.SetValue(clone, clonedPlanets);

        LastUpdatedField?.SetValue(clone, LastUpdatedField.GetValue(player));

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
        var getter = AccessorCache.GetGetter(obj.GetType(), name)!;
        return (T)getter(obj)!;
    }

    static void SetProperty(object obj, string name, object value)
    {
        var setter = AccessorCache.GetSetter(obj.GetType(), name)!;
        setter(obj, value);
    }

    static void CopyProperty(object source, object target, string name)
    {
        var getter = AccessorCache.GetGetter(source.GetType(), name);
        var setter = AccessorCache.GetSetter(target.GetType(), name);
        if (getter is null || setter is null)
        {
            return;
        }

        var value = getter(source);
        setter(target, value);
    }

    static class AccessorCache
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy;

        static readonly ConcurrentDictionary<(Type Type, string Name), Func<object, object?>> Getters = new();
        static readonly ConcurrentDictionary<(Type Type, string Name), Action<object, object?>> Setters = new();

        public static Func<object, object?>? GetGetter(Type type, string name)
            => Getters.GetOrAdd((type, name), key => BuildGetter(key.Type, key.Name));

        public static Action<object, object?>? GetSetter(Type type, string name)
            => Setters.GetOrAdd((type, name), key => BuildSetter(key.Type, key.Name));

        static Func<object, object?>? BuildGetter(Type type, string name)
        {
            var prop = type.GetProperty(name, Flags);
            var getter = prop?.GetGetMethod(true);
            if (prop is null || getter is null)
            {
                return null;
            }

            var objParam = Expression.Parameter(typeof(object), "obj");
            var castObj = Expression.Convert(objParam, type);
            var propAccess = Expression.Property(castObj, prop);
            var castResult = Expression.Convert(propAccess, typeof(object));
            return Expression.Lambda<Func<object, object?>>(castResult, objParam).Compile();
        }

        static Action<object, object?>? BuildSetter(Type type, string name)
        {
            var prop = type.GetProperty(name, Flags);
            var setter = prop?.GetSetMethod(true);
            if (prop is null || setter is null)
            {
                return null;
            }

            var objParam = Expression.Parameter(typeof(object), "obj");
            var valueParam = Expression.Parameter(typeof(object), "value");
            var castObj = Expression.Convert(objParam, type);
            var castValue = Expression.Convert(valueParam, prop.PropertyType);
            var body = Expression.Assign(Expression.Property(castObj, prop), castValue);
            return Expression.Lambda<Action<object, object?>>(body, objParam, valueParam).Compile();
        }
    }
}

