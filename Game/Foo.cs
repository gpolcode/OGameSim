using System;
using OGameSim.Entities;

namespace OGameSim.Production
{
    public static class Foo
    {
        public static float ApplyAction(Player player, long action)
        {
            float Penalty()
            {
                return -1;
            }

            float TryUpgrade(IUpgradable upgradable)
            {
                var currentPoints = player.Points;

                if (player.TrySpendResources(upgradable.UpgradeCost))
                {
                    upgradable.Upgrade();
                    var gainedPoints = (float)(player.Points - currentPoints);
                    return (float)Math.Log10(gainedPoints + 1);
                }

                return Penalty();
            }

            float ProceedToNextDay()
            {
                player.ProceedToNextDay();
                return 1;
            }

            var planetIndex = (int)Math.Floor(action / 3d) - 1;
            if (planetIndex > player.Planets.Count - 1)
            {
                return Penalty();
            }

            var reward = (action, action % 3) switch
            {
                (0, _) => ProceedToNextDay(),
                (1, _) => TryUpgrade(player.Astrophysics),
                (2, _) => TryUpgrade(player.PlasmaTechnology),
                (_, 0) => TryUpgrade(player.Planets[planetIndex].MetalMine),
                (_, 1) => TryUpgrade(player.Planets[planetIndex].CrystalMine),
                (_, 2) => TryUpgrade(player.Planets[planetIndex].DeuteriumSynthesizer),
                _ => throw new NotImplementedException(),
            };

            return reward;
        }

        public unsafe static void UpdateState(Player player, IntPtr statePointer)
        {
            var state = new Span<double>(statePointer.ToPointer(), 617);

            var currentIndex = 0;
            void SetStateValue(float value, Span<double> state)
            {
                state[currentIndex] = value;
                currentIndex++;
            }

            void AddResources(Resources resources, Span<double> state)
            {
                SetStateValue(resources.Metal, state);
                SetStateValue(resources.Crystal, state);
                SetStateValue(resources.Deuterium, state);
            }

            void AddResourcesModifier(ResourcesModifier resourcesModifier, Span<double> state)
            {
                SetStateValue((float)resourcesModifier.Metal, state);
                SetStateValue((float)resourcesModifier.Crystal, state);
                SetStateValue((float)resourcesModifier.Deuterium, state);
            }

            // Player
            AddResources(player.Resources, state);

            // Astro
            SetStateValue(player.Astrophysics.Level, state);
            AddResources(player.Astrophysics.UpgradeCost, state);

            // Plasma
            SetStateValue(player.PlasmaTechnology.Level, state);
            AddResources(player.PlasmaTechnology.UpgradeCost, state);
            AddResourcesModifier(player.PlasmaTechnology.Modifier, state);
            AddResourcesModifier(player.PlasmaTechnology.UpgradedModifier, state);

            // Planets
            foreach (var planet in player.Planets)
            {
                // Metal
                SetStateValue(planet.MetalMine.Level, state);
                AddResources(planet.MetalMine.UpgradeCost, state);
                AddResources(planet.MetalMine.TodaysProduction, state);
                AddResources(planet.MetalMine.UpgradeIncreasePerDay, state);

                // Crystal
                SetStateValue(planet.CrystalMine.Level, state);
                AddResources(planet.CrystalMine.UpgradeCost, state);
                AddResources(planet.CrystalMine.TodaysProduction, state);
                AddResources(planet.CrystalMine.UpgradeIncreasePerDay, state);

                // Deut
                SetStateValue(planet.DeuteriumSynthesizer.Level, state);
                AddResources(planet.DeuteriumSynthesizer.UpgradeCost, state);
                AddResources(planet.DeuteriumSynthesizer.TodaysProduction, state);
                AddResources(planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, state);
            }
        }
    }
}