using BranchBoundPlayer;
using OGameSim.Entities;
using Xunit;

public class BranchBoundPlayerTests
{
    [Fact]
    public void Evaluate_PrunesWhenBoundTooLow()
    {
        var player = new Player();
        decimal best = decimal.MaxValue;
        var result = Planner.Evaluate(player, 0, 1, ref best);
        Assert.Equal(decimal.MinValue, result);
    }
}
