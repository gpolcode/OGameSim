using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public sealed class MetalMine : Mine
    {
        public MetalMine()
            : base(new(30 * 24, 0, 0)) { }

        protected override Resources CalculateUpgradeCost()
        {
            var metalCost = 60 * Math.Pow(1.5, Level);
            var crystalCost = 15 * Math.Pow(1.5, Level);
            return new((ulong)Math.Floor(metalCost), (ulong)Math.Floor(crystalCost), 0);
        }

        protected override Resources CalculateUpgradedProduction()
        {
            var nextLevel = Level + 1;
            var productionPerHour = 30 * nextLevel * Math.Pow(1.1, nextLevel);
            var productionPerDay = (ulong)Math.Round(productionPerHour);
            productionPerDay *= 24;
            return new(productionPerDay, 0, 0);
        }
    }
}
