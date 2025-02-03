using OGameSim.Entities;
using OGameSim.Models;
using Xunit;

namespace Tests
{
    public class DeuteriumSynthesizerTests
    {
        [Theory]
        [InlineData(0, 0, 120)]
        [InlineData(0, 0, 0)]
        [InlineData(0, 0, -120)]
        [InlineData(1, 240, 120)]
        [InlineData(1, 360, 0)]
        [InlineData(1, 504, -120)]
        [InlineData(10, 5952, 120)]
        [InlineData(10, 8952, 0)]
        [InlineData(10, 11928, -120)]
        [InlineData(20, 30984, 120)]
        [InlineData(20, 46488, 0)]
        [InlineData(20, 61992, -120)]
        [InlineData(42, 529920, 120)]
        [InlineData(42, 794904, 0)]
        [InlineData(42, 1059864, -120)]
        public void Deuterium_synthesizer_should_produce_deuterium(
            uint level,
            ulong expectedProduction,
            int temperature
        )
        {
            // Setup
            var state = new GameState { AreResourcesFrozen = true };
            var planet = new Planet(temperature, 1);
            var subject = new DeuteriumSynthesizer(planet);
            for (var i = 0; i < level; i++)
            {
                subject.GetUpgrade().Apply(state);
            }

            // Act
            var production = subject.GetTodaysProduction();

            // Assert
            Assert.Equal(0u, production.Metal);
            Assert.Equal(0u, production.Crystal);
            Assert.Equal(expectedProduction, production.Deuterium);
        }

        [Theory]
        [InlineData(0, 225, 75, 0)]
        [InlineData(9, 8650, 2883, 0)]
        [InlineData(19, 498789, 166263, 0)]
        [InlineData(41, 3731849658, 1243949886, 0)]
        public void Deuterium_synthesizer_upgrade_should_cost(
            uint level,
            ulong metal,
            ulong crystal,
            ulong deuterium
        )
        {
            // Setup
            var state = new GameState { AreResourcesFrozen = true };
            var planet = new Planet(0, 1);
            var subject = new DeuteriumSynthesizer(planet);
            for (var i = 0; i < level; i++)
            {
                subject.GetUpgrade().Apply(state);
            }

            // Act
            var upgradeCost = subject.GetUpgrade().Cost;

            // Assert
            Assert.Equal(metal, upgradeCost.Metal);
            Assert.Equal(crystal, upgradeCost.Crystal);
            Assert.Equal(deuterium, upgradeCost.Deuterium);
        }
    }
}
