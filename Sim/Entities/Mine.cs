using OGameSim.Models;

namespace OGameSim.Entities
{
	public abstract class Mine : IUpgradeable, IResourceProducer
	{
		public uint Level { get; private set; }
		private Resources _production;

		protected Mine(Resources baseProduction)
		{
			_production = baseProduction;
		}

		public void Apply(Upgrade upgrade)
		{
			Level++;
			_production += upgrade.ProductionIncreasePerDay;
		}

		public Resources GetTodaysProduction()
		{
			return _production;
		}

		public Upgrade GetUpgrade()
		{
			var productionDelta = GetUpgradedProduction() - GetTodaysProduction();
			return new(GetUpgradeCost(), productionDelta, this);
		}

		protected abstract Resources GetUpgradeCost();

		protected abstract Resources GetUpgradedProduction();
	}
}