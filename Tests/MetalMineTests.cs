using OGameSim.Entities;
using Xunit;

namespace Tests
{
    public sealed class MetalMineTests
    {
        [Theory]
        [InlineData(0, 720)]
        [InlineData(1, 792)]
        [InlineData(10, 18672)]
        [InlineData(30, 376896)]
        [InlineData(50, 4226064)]
        public void Metal_mine_should_produce_metal(uint level, ulong expectedProduction)
        {
            // Setup
            var subject = new MetalMine();
            for (var i = 0; i < level; i++)
            {
                subject.Upgrade();
            }

            // Act
            var production = subject.TodaysProduction;

            // Assert
            Assert.Equal(expectedProduction, production.Metal);
            Assert.Equal(0u, production.Crystal);
            Assert.Equal(0u, production.Deuterium);
        }

        [Theory]
        [InlineData(0, 60, 15, 0)]
        [InlineData(9, 2306, 576, 0)]
        [InlineData(29, 7670042, 1917510, 0)]
        [InlineData(49, 25504860008, 6376215002, 0)]
        public void Metal_mine_upgrade_should_cost(
            uint level,
            ulong metal,
            ulong crystal,
            ulong deuterium
        )
        {
            // Setup
            var subject = new MetalMine();
            for (var i = 0; i < level; i++)
            {
                subject.Upgrade();
            }

            // Act
            var upgradeCost = subject.UpgradeCost;

            // Assert
            Assert.Equal(metal, upgradeCost.Metal);
            Assert.Equal(crystal, upgradeCost.Crystal);
            Assert.Equal(deuterium, upgradeCost.Deuterium);
        }
    }
}
