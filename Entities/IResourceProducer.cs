using OGameSim.Models;

namespace OGameSim.Entities
{
	public interface IResourceProducer
	{
		Resources GetTodaysProduction();
	}
}