using OGameSim.Entities;
using OGameSim.Production;
using PlanningPlayer;
using Xunit;
using System.Reflection;

namespace Tests
{
    public sealed class PlannerTests
    {
        [Fact]
        public void Player_clone_should_copy_properties()
        {
            // Setup
            var subject = new Player();
            subject.AddResources(new Resources(1000, 1000, 1000));
            subject.ProceedToNextDay();

            // Act
            var clone = subject.DeepClone();

            // Assert
            Assert.Equal(subject.Day, clone.Day);
            Assert.Equal(subject.Points, clone.Points);
            Assert.Equal(subject.Resources.Metal, clone.Resources.Metal);
            Assert.Equal(subject.Resources.Crystal, clone.Resources.Crystal);
            Assert.Equal(subject.Resources.Deuterium, clone.Resources.Deuterium);
            Assert.Equal(subject.Astrophysics.Level, clone.Astrophysics.Level);
            Assert.Equal(subject.PlasmaTechnology.Level, clone.PlasmaTechnology.Level);
            Assert.Equal(subject.Planets.Count, clone.Planets.Count);
            Assert.Equal(subject.Planets[0].MetalMine.Level, clone.Planets[0].MetalMine.Level);
            Assert.Equal(subject.Planets[0].CrystalMine.Level, clone.Planets[0].CrystalMine.Level);
            Assert.Equal(subject.Planets[0].DeuteriumSynthesizer.Level, clone.Planets[0].DeuteriumSynthesizer.Level);
        }

        [Fact]
        public void Player_clone_should_be_independent()
        {
            // Setup
            var subject = new Player();

            // Act
            var clone = subject.DeepClone();
            clone.AddResources(new Resources(1000, 1000, 1000));
            clone.Planets[0].MetalMine.Upgrade();
            clone.Astrophysics.Upgrade();
            clone.PlasmaTechnology.Upgrade();
            clone.ProceedToNextDay();

            // Assert
            Assert.Equal(0u, subject.Day);
            Assert.Equal(1u, clone.Day);
            Assert.Equal(0u, subject.Planets[0].MetalMine.Level);
            Assert.Equal(1u, clone.Planets[0].MetalMine.Level);
            Assert.Equal(1, subject.Planets.Count);
            Assert.Equal(2, clone.Planets.Count);
            Assert.Equal(0u, subject.Astrophysics.Level);
            Assert.Equal(1u, clone.Astrophysics.Level);
            Assert.Equal(0u, subject.PlasmaTechnology.Level);
            Assert.Equal(1u, clone.PlasmaTechnology.Level);
            Assert.Equal(0m, subject.Resources.Metal);
            Assert.NotEqual(subject.Resources.Metal, clone.Resources.Metal);
        }

        [Fact]
        public void Planner_calculate_roi_should_match_ratio()
        {
            // Setup
            var cost = new Resources(100, 0, 0);
            var gain = new Resources(10, 0, 0);

            // Act
            var roi = Planner.CalculateRoi(cost, gain);

            // Assert
            Assert.Equal(10, roi);
        }

        [Fact]
        public void Planner_search_should_return_current_points_when_horizon_zero()
        {
            var subject = new Player();
            var result = Planner.Search(subject, 0);
            Assert.Equal(subject.Points, result);
        }

        [Fact]
        public void Planner_enumerate_actions_should_not_throw()
        {
            var subject = new Player();
            var ex = Record.Exception(() => Planner.EnumerateActions(subject, 10));
            Assert.Null(ex);
        }

        [Fact]
        public void Planner_enumerate_actions_should_prune_by_payback()
        {
            var subject = new Player();
            var actions = Planner.EnumerateActions(subject, remainingDays: 0);
            Assert.Single(actions);
            Assert.Null(actions[0].Upgradable);
        }

        [Fact]
        public void Planner_build_key_should_bucket_resources()
        {
            var p1 = new Player();
            p1.AddResources(new Resources(500, 0, 0));
            var p2 = new Player();
            p2.AddResources(new Resources(999, 0, 0));
            var method = typeof(Planner).GetMethod("BuildKey", BindingFlags.NonPublic | BindingFlags.Static)!;
            var key1 = method.Invoke(null, new object?[] { p1, 1000, 50 }) as string;
            var key2 = method.Invoke(null, new object?[] { p2, 1000, 50 }) as string;
            Assert.Equal(key1, key2);
        }

        [Fact]
        public void Planner_search_with_buckets_should_match_exact_for_small_horizon()
        {
            var player = new Player();
            var exact = Planner.Search(player, horizon: 2, bucketSize: 1, bucketCount: int.MaxValue);
            var bucketed = Planner.Search(player, horizon: 2, bucketSize: 1000, bucketCount: 50);
            Assert.Equal(exact, bucketed);
        }
    }
}
