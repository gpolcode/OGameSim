using OGameSim.Entities;
using Xunit;

namespace Tests
{
    public class PlayerTests
    {
        [Theory]
        [InlineData(0, 1)]
        [InlineData(1, 2)]
        [InlineData(2, 2)]
        [InlineData(3, 3)]
        [InlineData(13, 8)]
        [InlineData(14, 8)]
        [InlineData(22, 12)]
        [InlineData(23, 13)]
        [InlineData(30, 16)]
        [InlineData(31, 17)]
        public void Player_should_have_planet_count(uint astrophysicsLevel, int planetCount)
        {
            // Setup
            var subject = new Player();
            for (var i = 0; i < astrophysicsLevel; i++)
            {
                subject.Astrophysics.Upgrade();
            }

            // Act
            var planets = subject.Planets;

            // Assert
            Assert.Equal(planetCount, planets.Count);
        }

        [Fact]
        public void Player_should_have_no_resources()
        {
            // Setup
            var subject = new Player();

            // Act
            var resources = subject.Resources;

            // Assert
            Assert.Equal(0m, resources.Metal);
            Assert.Equal(0m, resources.Crystal);
            Assert.Equal(0m, resources.Deuterium);
        }

        [Fact]
        public void Player_should_gain_resources()
        {
            // Setup
            var subject = new Player();

            // Act
            subject.Planets[0].DeuteriumSynthesizer.Upgrade();
            subject.ProceedToNextDay();
            var resources = subject.Resources;

            // Assert
            Assert.Equal(1440m, resources.Metal);
            Assert.Equal(720m, resources.Crystal);
            Assert.Equal(960m, resources.Deuterium);
        }

        [Fact]
        public void Player_should_spend_resources()
        {
            // Setup
            var subject = new Player();
            subject.AddResources(new(10, 20, 30)); // 140 MSU

            // Act
            var result = subject.TrySpendResources(new(1, 2, 3)); // 14 MSU
            var resources = subject.Resources;

            // Assert
            Assert.True(result);
            Assert.Equal(126m, resources.Metal);
            Assert.Equal(0m, resources.Crystal);
            Assert.Equal(0m, resources.Deuterium);
        }

        [Fact]
        public void Player_should_spend_exact_resources()
        {
            // Setup
            var subject = new Player();
            subject.AddResources(new(1, 2, 3)); // 14 MSU

            // Act
            var result = subject.TrySpendResources(new(1, 2, 3)); // 14 MSU
            var resources = subject.Resources;

            // Assert
            Assert.True(result);
            Assert.Equal(0m, resources.Metal);
            Assert.Equal(0m, resources.Crystal);
            Assert.Equal(0m, resources.Deuterium);
        }

        [Fact]
        public void Player_should_not_overspend_resources()
        {
            // Setup
            var subject = new Player();
            subject.AddResources(new(1, 2, 3)); // 14 MSU

            // Act
            var result = subject.TrySpendResources(new(10, 20, 30)); // 140 MSU
            var resources = subject.Resources;

            // Assert
            Assert.False(result);
            Assert.Equal(1m, resources.Metal);
            Assert.Equal(2m, resources.Crystal);
            Assert.Equal(3m, resources.Deuterium);
        }
    }
}
