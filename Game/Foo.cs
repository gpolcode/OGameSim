using System;
using System.Linq;
using OGameSim.Entities;

namespace OGameSim.Production
{
    public static class Foo
    {

        public record PlayerStats(float MetalMax, float MetalAverage, float MetalMin, float CrystalMax, float CrystalAverage, float CrystalMin, float DeutMax, float DeutAverage, float DeutMin)
        {
        }

        public static PlayerStats GetPlayerStats(Player player)
        {
            var metalLevels = player.Planets.Select(x => (float)x.MetalMine.Level);
            var crystalLevels = player.Planets.Select(x => (float)x.CrystalMine.Level);
            var deutLevels = player.Planets.Select(x => (float)x.DeuteriumSynthesizer.Level);

            return new(metalLevels.Max(), metalLevels.Average(), metalLevels.Min(),
                crystalLevels.Max(), crystalLevels.Average(), crystalLevels.Min(),
                deutLevels.Max(), deutLevels.Average(), deutLevels.Min());
        }

        public static (float, bool) ApplyAction(Player player, long action)
        {
            (float, bool) Penalty()
            {
                return (-1, false);
            }

            (float, bool) Terminate()
            {
                return (0, true);
            }

            (float, bool) TryUpgrade(IUpgradable upgradable)
            {
                var currentPoints = player.Points;

                if (player.TrySpendResources(upgradable.UpgradeCost))
                {
                    upgradable.Upgrade();
                    var gainedPoints = (float)(player.Points - currentPoints);
                    return ((float)Math.Log10(gainedPoints + 1), false);
                }

                return Terminate();
            }

            (float, bool) ProceedToNextDay()
            {
                player.ProceedToNextDay();
                return (0, false);
            }

            var planetIndex = (int)Math.Floor(action / 3d) - 1;
            if (planetIndex > player.Planets.Count - 1)
            {
                return Terminate();
            }

            var result = (action, action % 3) switch
            {
                (0, _) => ProceedToNextDay(),
                (1, _) => TryUpgrade(player.Astrophysics),
                (2, _) => TryUpgrade(player.PlasmaTechnology),
                (_, 0) => TryUpgrade(player.Planets[planetIndex].MetalMine),
                (_, 1) => TryUpgrade(player.Planets[planetIndex].CrystalMine),
                (_, 2) => TryUpgrade(player.Planets[planetIndex].DeuteriumSynthesizer),
                _ => throw new NotImplementedException(),
            };

            return result;
        }

        public unsafe static void UpdateState(Player player, IntPtr statePointer)
        {
            var state = new Span<double>(statePointer.ToPointer(), 65);

            var currentIndex = 0;
            var todaysProduction = player.GetTodaysProduction();
            void SetStateValue(float value, Span<double> state)
            {
                state[currentIndex] = value;
                currentIndex++;
            }

            void AddResources(Resources resources, Span<double> state)
            {
                SetStateValue(resources.ConvertToMetalValue(), state);
            }

            // Player
            AddResources(player.Resources, state);
            AddResources(todaysProduction, state);

            // Astro
            AddResources(player.Astrophysics.UpgradeCost, state);

            // Plasma
            AddResources(player.PlasmaTechnology.UpgradeCost, state);
            AddResources(todaysProduction * (player.PlasmaTechnology.UpgradedModifier - player.PlasmaTechnology.Modifier), state);

            // Planets
            foreach (var planet in player.Planets)
            {
                // Metal
                AddResources(planet.MetalMine.UpgradeCost, state);
                AddResources(planet.MetalMine.UpgradeIncreasePerDay, state);

                // Crystal
                AddResources(planet.CrystalMine.UpgradeCost, state);
                AddResources(planet.CrystalMine.UpgradeIncreasePerDay, state);

                // Deut
                AddResources(planet.DeuteriumSynthesizer.UpgradeCost, state);
                AddResources(planet.DeuteriumSynthesizer.UpgradeIncreasePerDay, state);
            }
        }
    }
}