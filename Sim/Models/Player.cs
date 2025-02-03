using System.Collections.Generic;
using OGameSim.Entities;

namespace OGameSim.Models
{
    public class Player
    {
        private List<Planet> _planets = [];
        public IReadOnlyList<Planet> Planets
        {
            get
            {
                var missingPlanets = (Astrophysics.Level / 2) + 1 - _planets.Count;
                for (int i = 0; i < missingPlanets; i++)
                {
                    _planets.Add(new(-115));
                }

                return _planets;
            }
        }

        public Astrophysics Astrophysics { get; } = new();
        public PlasmaTechnology PlasmaTechnology { get; } = new();

        public decimal Points { get; private set; }
        public Resources Resources { get; private set; }

        private void AddResources(Resources resources)
        {
            Resources += resources;
        }

        public bool TrySpendResources(Resources resourcesToSpend)
        {
            if (!Resources.CanSubtract(resourcesToSpend))
            {
                return false;
            }

            var resourcesValue = Resources.ConvertToMetalValue();
            var resourcesToSpendValue = resourcesToSpend.ConvertToMetalValue();
            Resources = new(resourcesValue - resourcesToSpendValue, 0, 0);

            Points += resourcesToSpend.Metal / 1000m;
            Points += resourcesToSpend.Crystal / 1000m;
            Points += resourcesToSpend.Deuterium / 1000m;

            return true;
        }

        public void ProceedToNextDay()
        {
            Resources mineProduction = new();
            foreach (var planet in Planets)
            {
                mineProduction += planet.MetalMine.TodaysProduction;
                mineProduction += planet.CrystalMine.TodaysProduction;
                mineProduction += planet.DeuteriumSynthesizer.TodaysProduction;
            }

            var plasmaTechnologyBonus = mineProduction * PlasmaTechnology.Modifier;
            AddResources(mineProduction);
            AddResources(plasmaTechnologyBonus);
        }
    }
}
