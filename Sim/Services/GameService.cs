using OGameSim.Models;

namespace OGameSim.Services
{
	public class GameService
	{
		private readonly GameState _state;

		public GameService(GameState state)
		{
			_state = state;
		}

		public void MoveToNextDay()
		{
			Resources mineProduction = new();
			foreach (var planet in _state.Planets)
			{
				mineProduction += planet.MetalMine.GetTodaysProduction();
				mineProduction += planet.CrystalMine.GetTodaysProduction();
				mineProduction += planet.DeuteriumSynthesizer.GetTodaysProduction();
			}

			_state.AddResources(mineProduction);
		}
	}
}