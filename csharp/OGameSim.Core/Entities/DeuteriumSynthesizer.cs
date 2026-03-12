using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public sealed class DeuteriumSynthesizer(int planetMaxTemperature) : Mine(new())
    {
        private readonly int _planetAverageTemperature = planetMaxTemperature - 20;

        protected override Resources CalculateUpgradeCost()
        {
            var metalCost = 225 * Math.Pow(1.5, Level);
            var crystalCost = 75 * Math.Pow(1.5, Level);
            return new((ulong)Math.Round(metalCost), (ulong)Math.Round(crystalCost), 0);
        }

        protected override Resources CalculateUpgradedProduction()
        {
            var nextLevel = Level + 1;
            var productionPerHour =
                20
                * nextLevel
                * Math.Pow(1.1, nextLevel)
                * (0.68 - 0.002 * _planetAverageTemperature);

            var productionPerDay = (ulong)Math.Floor(productionPerHour);
            productionPerDay *= 24;
            return new(0, 0, productionPerDay);
        }
    }
}
