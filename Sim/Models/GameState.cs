using System.Collections.Generic;

namespace OGameSim.Models
{
	public class GameState
	{
		public bool AreResourcesFrozen { get; set; }
		public List<Planet> Planets { get; } = new();

		//public Astrophysics Astrophysics { get; }
		//public PlasmaTechnic PlasmaTechnic { get; }

		public double Points { get; private set; }
		public Resources Resources { get; private set; }

		public void AddResources(Resources resources)
		{
			Resources += resources;
		}

		public void SpendResources(Resources resourcesToSpend)
		{
			if (!AreResourcesFrozen)
			{
				var resourcesValue = Resources.ConvertToMetalValue();
				var resourcesToSpendValue = resourcesToSpend.ConvertToMetalValue();
				Resources = new(resourcesValue - resourcesToSpendValue, 0, 0);
			}

			Points += resourcesToSpend.Metal / 1000d;
			Points += resourcesToSpend.Crystal / 1000d;
			Points += resourcesToSpend.Deuterium / 1000d;
		}
	}
}