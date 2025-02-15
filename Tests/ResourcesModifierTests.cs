using OGameSim.Production;
using Xunit;

namespace Tests
{
    public sealed class ResourcesModifierTests
    {
        [Fact]
        public void Minus_operator_should_reduce_modifier()
        {
            // Setup
            var subject = new ResourcesModifier(23, 59, 131);
            var other = new ResourcesModifier(2, 7, 13);

            // Act
            var result = subject - other;

            // Assert
            Assert.Equal(21u, result.Metal);
            Assert.Equal(52u, result.Crystal);
            Assert.Equal(118u, result.Deuterium);
        }
    }
}
