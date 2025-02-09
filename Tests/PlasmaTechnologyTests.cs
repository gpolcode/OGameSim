using OGameSim.Entities;
using Xunit;

namespace Tests
{
    public sealed class PlasmaTechnologyTests
    {
        [Theory]
        [InlineData(0, 0, 0, 0)]
        [InlineData(1, 0.01, 0.0066, 0.0033)]
        [InlineData(9, 0.09, 0.0594, 0.0297)]
        [InlineData(15, 0.15, 0.099, 0.0495)]
        [InlineData(20, 0.2, 0.132, 0.066)]
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
        [InlineData(0, 0.01, 0.0066, 0.0033)]
        [InlineData(8, 0.09, 0.0594, 0.0297)]
        [InlineData(14, 0.15, 0.099, 0.0495)]
        [InlineData(19, 0.2, 0.132, 0.066)]
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
