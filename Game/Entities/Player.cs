using System;
using System.Collections.Generic;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public sealed class Player
    {
        private uint? _lastUpdatedAstroLevel;
        private List<Planet> _planets = [];
        public IReadOnlyList<Planet> Planets
        {
            get
            {
                if (Astrophysics.Level != _lastUpdatedAstroLevel)
                {
                    var missingPlanets = Math.Ceiling(Astrophysics.Level / 2d) + 1 - _planets.Count;
                    for (int i = 0; i < missingPlanets; i++)
                    {
                        _planets.Add(new(-115));
                    }

                    _lastUpdatedAstroLevel = Astrophysics.Level;
                }

                return _planets;
            }
        }

        public Astrophysics Astrophysics { get; } = new();
        public PlasmaTechnology PlasmaTechnology { get; } = new();

        public decimal Points { get; private set; }
        public uint Day { get; private set; }
        public Resources Resources { get; private set; }

        internal void AddResources(Resources resources)
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
            Day++;
            Resources mineProduction = new();
            foreach (var planet in Planets)
            {
                mineProduction += planet.MetalMine.TodaysProduction;
                mineProduction += planet.CrystalMine.TodaysProduction;
                mineProduction += planet.DeuteriumSynthesizer.TodaysProduction;
            }

            Resources modifierProduction = new();
            var modifiers = new ResourcesModifier[] { PlasmaTechnology.Modifier };
            foreach (var modifier in modifiers)
            {
                modifierProduction += mineProduction * modifier;
            }

            AddResources(mineProduction);
            AddResources(modifierProduction);
        }
    }
}
