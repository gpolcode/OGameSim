using OGameSim.Models;

namespace OGameSim.Entities
{
	public interface IUpgradeable
	{
		void Apply(Upgrade upgrade);
		Upgrade GetUpgrade();
	}
}