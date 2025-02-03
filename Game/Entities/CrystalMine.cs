using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public class CrystalMine : Mine
    {
        public CrystalMine()
            : base(new(0, 15 * 24, 0)) { }

        protected override Resources CalculateUpgradeCost()
        {
            var metalCost = 48 * Math.Pow(1.6, Level);
            var crystalCost = 24 * Math.Pow(1.6, Level);
            return new((ulong)Math.Ceiling(metalCost), (ulong)Math.Ceiling(crystalCost), 0);
        }

        protected override Resources CalculateUpgradedProduction()
        {
            var nextLevel = Level + 1;
            var productionPerHour = 20 * nextLevel * Math.Pow(1.1, nextLevel);
            var productionPerDay = (ulong)Math.Floor(productionPerHour);
            productionPerDay *= 24;
            return new(0, productionPerDay, 0);
        }
    }
}
