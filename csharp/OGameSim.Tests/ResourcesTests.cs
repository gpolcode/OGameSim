using OGameSim.Production;
using Xunit;

namespace Tests
{
    public sealed class ResourcesTests
    {
        [Theory]
        [InlineData(10, 0, 0, 20, 0, 0)]
        [InlineData(0, 10, 0, 0, 20, 0)]
        [InlineData(0, 0, 10, 0, 0, 20)]
        public void CanSubtract_should_be_false(
            ulong metal,
            ulong crystal,
            ulong deuterium,
            ulong otherMetal,
            ulong otherCrystal,
            ulong otherDeuterium
        )
        {
            // Setup
            var subject = new Resources(metal, crystal, deuterium);
            var other = new Resources(otherMetal, otherCrystal, otherDeuterium);

            // Act
            var canSubtract = subject.CanSubtract(other);

            // Assert
            Assert.False(canSubtract);
        }

        [Theory]
        [InlineData(10, 0, 0, 10, 0, 0)]
        [InlineData(0, 10, 0, 0, 10, 0)]
        [InlineData(0, 0, 10, 0, 0, 10)]
        public void CanSubtract_should_be_true(
            ulong metal,
            ulong crystal,
            ulong deuterium,
            ulong otherMetal,
            ulong otherCrystal,
            ulong otherDeuterium
        )
        {
            // Setup
            var subject = new Resources(metal, crystal, deuterium);
            var other = new Resources(otherMetal, otherCrystal, otherDeuterium);

            // Act
            var canSubtract = subject.CanSubtract(other);

            // Assert
            Assert.True(canSubtract);
        }

        [Fact]
        public void Minus_operator_should_remove_resources()
        {
            // Setup
            var subject = new Resources(23, 59, 131);
            var other = new Resources(2, 7, 13);

            // Act
            var result = subject - other;

            // Assert
            Assert.Equal(21u, result.Metal);
            Assert.Equal(52u, result.Crystal);
            Assert.Equal(118u, result.Deuterium);
        }

        [Fact]
        public void Plus_operator_should_add_resources()
        {
            // Setup
            var subject = new Resources(23, 59, 131);
            var other = new Resources(2, 7, 13);

            // Act
            var result = subject + other;

            // Assert
            Assert.Equal(25u, result.Metal);
            Assert.Equal(66u, result.Crystal);
            Assert.Equal(144u, result.Deuterium);
        }

        [Fact]
        public void Star_operator_should_multiply_resources()
        {
            // Setup
            var subject = new Resources(23, 59, 131);
            var modifier = new ResourcesModifier(2, 7, 13);

            // Act
            var result = subject * modifier;

            // Assert
            Assert.Equal(46u, result.Metal);
            Assert.Equal(413u, result.Crystal);
            Assert.Equal(1703u, result.Deuterium);
        }
    }
}
