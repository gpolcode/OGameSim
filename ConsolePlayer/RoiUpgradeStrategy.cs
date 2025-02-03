using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using OGameSim.Entities;

namespace OGameSim.Services
{
    public class RoiUpgradeStrategy
    {
        public Player Player { get; } = new();

        public double CalculateRoi(Upgrade upgrade)
        {
            var weightedCost = upgrade.Cost.ConvertToMetalValue();
            var weightedIncrease = upgrade.ProductionIncreasePerDay.ConvertToMetalValue();
            return weightedCost / weightedIncrease;
        }

        public void FindAndBuildUpgrades()
        {
            var upgradeables = GetUpgradeables();

            while (TryGetUpgrade(upgradeables, out var upgrade))
            {
                upgrade.Apply(_state);
            }
        }

        public List<IUpgradeable> GetUpgradeables()
        {
            var upgradeables = new List<IUpgradeable>();
            foreach (var planet in _state.Planets)
            {
                upgradeables.Add(planet.MetalMine);
                upgradeables.Add(planet.CrystalMine);
                upgradeables.Add(planet.DeuteriumSynthesizer);
            }

            return upgradeables;
        }

        public bool TryGetUpgrade(
            List<IUpgradeable> upgradeables,
            [NotNullWhen(true)] out Upgrade? foundUpgrade
        )
        {
            var upgrades = new List<Upgrade>();
            for (var i = upgradeables.Count - 1; i >= 0; i--)
            {
                var upgrade = upgradeables[i].GetUpgrade();
                if (_state.Resources.CanSubtract(upgrade.Cost))
                {
                    upgrades.Add(upgrade);
                }
                else
                {
                    upgradeables.RemoveAt(i);
                }
            }

            if (upgrades.Any())
            {
                foundUpgrade = upgrades
                    .ToLookup(CalculateRoi)
                    .OrderByDescending(x => x.Key)
                    .First()
                    .First();

                return true;
            }
            else
            {
                foundUpgrade = null;
                return false;
            }
        }
    }
}
