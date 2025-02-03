using System;

namespace OGameSim.Models
{
    public readonly record struct Resources(ulong Metal, ulong Crystal, ulong Deuterium)
    {
        public static Resources operator +(Resources a, Resources b) =>
            new(a.Metal + b.Metal, a.Crystal + b.Crystal, a.Deuterium + b.Deuterium);

        public static Resources operator -(Resources a, Resources b) =>
            new(a.Metal - b.Metal, a.Crystal - b.Crystal, a.Deuterium - b.Deuterium);

        public static Resources operator *(Resources a, ResourcesModifier b) =>
            new(
                (ulong)Math.Floor(a.Metal * b.Metal),
                (ulong)Math.Floor(a.Crystal * b.Crystal),
                (ulong)Math.Floor(a.Deuterium * b.Deuterium)
            );

        public bool CanSubtract(Resources other)
        {
            return ConvertToMetalValue() >= other.ConvertToMetalValue();
        }

        public ulong ConvertToMetalValue()
        {
            var metalValue = Metal * ResourceWeight.MetalValue;
            var crystalValue = Crystal * ResourceWeight.CrystalValue;
            var deuteriumValue = Deuterium * ResourceWeight.DeuteriumValue;

            return metalValue + crystalValue + deuteriumValue;
        }
    }
}
