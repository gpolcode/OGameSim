using OGameSim.Entities;
using Xunit;

namespace Tests
{
    public sealed class AstrophysicsTests
    {
        [Theory]
        [InlineData(0, 4000, 8000, 4000)]
        [InlineData(1, 7000, 14000, 7000)]
        [InlineData(2, 12250, 24500, 12250)]
        [InlineData(3, 21437, 42875, 21437)]
        [InlineData(8, 351855, 703711, 351855)]
        [InlineData(14, 10106311, 20212622, 10106311)]
        [InlineData(24, 2722533045, 5445066090, 2722533045)]
        [InlineData(30, 78199045470, 156398090941, 78199045470)]
        public void Astrophysics_upgrade_should_cost(
            uint level,
            ulong metal,
            ulong crystal,
            ulong deuterium
        )
        {
            // Setup
            var subject = new Astrophysics();
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
