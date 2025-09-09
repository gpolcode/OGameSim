using OGameSim.Entities;
using OGameSim.Production;
using PlanningPlayer;
using Xunit;

namespace Tests;

public sealed class PlannerTests
{
    [Fact]
    public void CloneProducesIndependentCopy()
    {
        var player = new Player();
        var clone = player.DeepClone();
        clone.ProceedToNextDay();
        Assert.NotEqual(player.Day, clone.Day);
    }

    [Fact]
    public void CalculateRoiMatchesRatio()
    {
        var cost = new Resources(100, 0, 0);
        var gain = new Resources(10, 0, 0);
        var roi = Planner.CalculateRoi(cost, gain);
        Assert.Equal(10, roi);
    }

    [Fact]
    public void SearchWithZeroHorizonReturnsCurrentPoints()
    {
        var player = new Player();
        var result = Planner.Search(player, 0);
        Assert.Equal(player.Points, result);
    }
}
