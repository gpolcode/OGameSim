using OGameSim.Models;

namespace OGameSim.Services
{
	public class InitialGameStateCreator
	{
		//public ulong CrystalMineLevel { get; set; }
		//public ulong DeuteriumMineLevel { get; set; }
		//public ulong MetalMineLevel { get; set; }
		public uint PlanetCount { get; set; }
		public int PlanetMaxTemperatur { get; set; }

		public uint PlanetPosition { get; set; }

		public uint SimulationDays { get; set; }
		//public Resources Resources { get; set; }

		public GameState Create()
		{
			var state = new GameState();

			for (var i = 0; i < PlanetCount; i++)
			{
				state.Planets.Add(new(PlanetMaxTemperatur, PlanetPosition));
			}

			return state;
		}
	}
}