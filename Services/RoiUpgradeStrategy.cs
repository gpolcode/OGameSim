using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;
using OGameSim.Models;

namespace OGameSim.Services
{
	public class RoiUpgradeStrategy : IUpgradeStrategy
	{
		private readonly GameState _state;

		public RoiUpgradeStrategy(GameState state)
		{
			_state = state;
		}

		public double CalculateRoi(Upgrade upgrade)
		{
			var weightedCost = upgrade.Cost.ConvertToMetalValue();
			var weightedIncrease = upgrade.ProductionIncreasePerDay.ConvertToMetalValue();
			return weightedCost / weightedIncrease;
		}

		public void FindAndBuildUpgrades()
		{
			bool hasFoundUpgrade;
			var upgradeables = GetUpgradeables();
			do
			{
				var bestUpgrade = TryFindBestUpgrade(upgradeables, out hasFoundUpgrade);
				if (hasFoundUpgrade)
				{
					bestUpgrade.Apply(_state);
				}
			} while (hasFoundUpgrade);
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

		public Upgrade? TryFindBestUpgrade(List<IUpgradeable> upgradeables, out bool hasFoundUpgrade)
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
				hasFoundUpgrade = true;
				return upgrades.ToLookup(CalculateRoi)
					.OrderByDescending(x => x.Key)
					.First()
					.First();
			}
			else
			{
				hasFoundUpgrade = false;
				return null;
			}
		}
	}
}