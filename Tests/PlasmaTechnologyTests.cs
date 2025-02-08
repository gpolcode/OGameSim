using OGameSim.Entities;
using Xunit;

namespace Tests
{
    public class PlasmaTechnologyTests
    {
        [Theory]
        [InlineData(0, 1, 1, 1)]
        [InlineData(1, 1.01, 1.0066, 1.0033)]
        [InlineData(9, 1.09, 1.0594, 1.0297)]
        [InlineData(15, 1.15, 1.099, 1.0495)]
        [InlineData(20, 1.2, 1.132, 1.066)]
        public void Plasma_should_modify(
            uint level,
            decimal metal,
            decimal crystal,
            decimal deuterium
        )
        {
            // Setup
            var subject = new PlasmaTechnology();
            for (var i = 0; i < level; i++)
            {
                subject.Upgrade();
            }

            // Act
            var modifier = subject.Modifier;

            // Assert
            Assert.Equal(metal, modifier.Metal);
            Assert.Equal(crystal, modifier.Crystal);
            Assert.Equal(deuterium, modifier.Deuterium);
        }

        [Theory]
        [InlineData(0, 1.01, 1.0066, 1.0033)]
        [InlineData(8, 1.09, 1.0594, 1.0297)]
        [InlineData(14, 1.15, 1.099, 1.0495)]
        [InlineData(19, 1.2, 1.132, 1.066)]
        public void Plasma_modifier_should_increase(
            uint level,
            decimal metal,
            decimal crystal,
            decimal deuterium
        )
        {
            // Setup
            var subject = new PlasmaTechnology();
            for (var i = 0; i < level; i++)
            {
                subject.Upgrade();
            }

            // Act
            var modifier = subject.UpgradedModifier;

            // Assert
            Assert.Equal(metal, modifier.Metal);
            Assert.Equal(crystal, modifier.Crystal);
            Assert.Equal(deuterium, modifier.Deuterium);
        }

        [Theory]
        [InlineData(0, 2000, 4000, 1000)]
        [InlineData(1, 4000, 8000, 2000)]
        [InlineData(9, 1024000, 2048000, 512000)]
        [InlineData(15, 65536000, 131072000, 32768000)]
        [InlineData(19, 1048576000, 2097152000, 524288000)]
        public void Plasma_upgrade_should_cost(
            uint level,
            ulong metal,
            ulong crystal,
            ulong deuterium
        )
        {
            // Setup
            var subject = new PlasmaTechnology();
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
