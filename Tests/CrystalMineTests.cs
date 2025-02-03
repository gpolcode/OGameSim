using OGameSim.Entities;
using OGameSim.Models;
using Xunit;

namespace Tests
{
    public class CrystalMineTests
    {
        [Theory]
        [InlineData(0, 360)]
        [InlineData(1, 528)]
        [InlineData(10, 12432)]
        [InlineData(20, 64560)]
        [InlineData(34, 416928)]
        public void Crystal_mine_should_produce_crystal(uint level, ulong expectedProduction)
        {
            // Setup
            var state = new GameState { AreResourcesFrozen = true };
            var subject = new CrystalMine();
            for (var i = 0; i < level; i++)
            {
                subject.GetUpgrade().Apply(state);
            }

            // Act
            var production = subject.GetTodaysProduction();

            // Assert
            Assert.Equal(0u, production.Metal);
            Assert.Equal(expectedProduction, production.Crystal);
            Assert.Equal(0u, production.Deuterium);
        }

        [Theory]
        [InlineData(0, 48, 24, 0)]
        [InlineData(9, 3299, 1650, 0)]
        [InlineData(19, 362678, 181339, 0)]
        [InlineData(33, 261336858, 130668429, 0)]
        public void Crystal_mine_upgrade_should_cost(
            uint level,
            ulong metal,
            ulong crystal,
            ulong deuterium
        )
        {
            // Setup
            var state = new GameState { AreResourcesFrozen = true };
            var subject = new CrystalMine();
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
