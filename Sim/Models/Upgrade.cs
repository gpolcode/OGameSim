using OGameSim.Entities;

namespace OGameSim.Models
{
	public class Upgrade
	{
		public Resources Cost { get; }
		public Resources ProductionIncreasePerDay { get; }
		public IUpgradeable Upgradeable { get; }

		public Upgrade(Resources cost, Resources productionIncreasePerDay, IUpgradeable upgradeable)
		{
			Cost = cost;
			ProductionIncreasePerDay = productionIncreasePerDay;
			Upgradeable = upgradeable;
		}

		public void Apply(GameState state)
		{
			state.SpendResources(Cost);
			Upgradeable.Apply(this);
		}
	}
}