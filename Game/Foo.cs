using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OGameSim.Entities;

namespace OGameSim.Production
{
    public static class Foo
    {
        public record PlayerStats(
            float MetalMax,
            float MetalAverage,
            float MetalMin,
            float CrystalMax,
            float CrystalAverage,
            float CrystalMin,
            float DeutMax,
            float DeutAverage,
            float DeutMin
        )
        { }

        public sealed class ExplorationReward
        {
            public ExplorationReward(float value)
            {
                Value = value;
            }

            public bool Redeemed { get; private set; }

            public bool TryClaim()
            {
                if (Redeemed)
                {
                    return false;
                }

                lock (this)
                {
                    if (Redeemed)
                    {
                        return false;
                    }
                    else
                    {
                        Redeemed = true;
                        return true;
                    }
                }
            }

            public float Value { get; }
        }

        public static PlayerStats GetPlayerStats(Player player)
        {
            var metalLevels = player.Planets.Select(x => (float)x.MetalMine.Level);
            var crystalLevels = player.Planets.Select(x => (float)x.CrystalMine.Level);
            var deutLevels = player.Planets.Select(x => (float)x.DeuteriumSynthesizer.Level);

            return new(
                metalLevels.Max(),
                metalLevels.Average(),
                metalLevels.Min(),
                crystalLevels.Max(),
                crystalLevels.Average(),
                crystalLevels.Min(),
                deutLevels.Max(),
                deutLevels.Average(),
                deutLevels.Min()
            );
        }

        private static readonly FrozenDictionary<decimal, ExplorationReward> _rewards;
        private const int RewardDistribution = 5_000_000;

        static Foo()
        {
            var maxValue = 25f;

            var bucketCount = 300_000_000 / RewardDistribution;
            var rewards = new Dictionary<decimal, ExplorationReward>();

            for (int i = 0; i < bucketCount; i++)
            {
                var points = i * RewardDistribution;
                var value = maxValue / bucketCount * i;
                rewards.Add(points, new(value));
            }

            _rewards = rewards.ToFrozenDictionary();
        }

        public static float GetExplorationReward(Player player)
        {
            var bucket = Math.Floor(player.Points / RewardDistribution) * RewardDistribution;
            var explorationReward = _rewards[bucket];
            return explorationReward.TryClaim() ? explorationReward.Value : 0f;
        }

        public static (float, bool) ApplyAction(Player player, long action)
        {
            (float, bool) Penalty()
            {
                return (-0.1f, false);
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
                    var upgradeReward = (float)Math.Log10(gainedPoints + 1);
                    var explorationReward = GetExplorationReward(player);

                    return (upgradeReward + explorationReward, false);
                }

                return Penalty();
            }

            (float, bool) ProceedToNextDay()
            {
                player.ProceedToNextDay();
                return (0.1f, false);
            }

            var planetIndex = (int)Math.Floor(action / 3d) - 1;
            if (planetIndex > player.Planets.Count - 1)
            {
                return Penalty();
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

        public static unsafe void UpdateState(Player player, IntPtr statePointer)
        {
            var state = new Span<double>(statePointer.ToPointer(), 125);

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
            AddResources(
                todaysProduction
                    * (player.PlasmaTechnology.UpgradedModifier - player.PlasmaTechnology.Modifier),
                state
            );

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

        public static unsafe (float[] Rewards, bool[] Terminated) StepBatch(
            Player[] players,
            long[] actions,
            double[] states
        )
        {
            const int stateSize = 125;
            if (players.Length != actions.Length)
            {
                throw new ArgumentException("players/actions length mismatch");
            }

            if (states.Length != players.Length * stateSize)
            {
                throw new ArgumentException("states length mismatch");
            }

            var rewards = new float[players.Length];
            var terminated = new bool[players.Length];

            fixed (double* statePtr = states)
            {
                var basePtr = (IntPtr)statePtr;
                Parallel.For(0, players.Length, i =>
                {
                    var result = ApplyAction(players[i], actions[i]);
                    rewards[i] = result.Item1;
                    terminated[i] = result.Item2;

                    var offsetPtr = IntPtr.Add(basePtr, i * stateSize * sizeof(double));
                    UpdateState(players[i], offsetPtr);
                });
            }

            return (rewards, terminated);
        }
    }
}
