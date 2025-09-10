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

    [Fact]
    public void Search_HorizonOne_NoPoints()
    {
        var player = new Player();
        var result = Planner.Search(player, 1);
        Assert.Equal(0m, result);
    }

    [Fact]
    public void Search_HorizonTwo_PerformsUpgrade()
    {
        var player = new Player();
        var result = Planner.Search(player, 2);
        Assert.Equal(0.300m, result);
    }
}
