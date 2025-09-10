using PaybackPruningPlayer;
using OGameSim.Entities;
using Xunit;

public class PaybackPruningPlayerTests
{
    [Fact]
    public void EnumerateActions_FiltersLongPayback()
    {
        var player = new Player();
        var actions = Planner.EnumerateActions(player, 0);
        Assert.Single(actions);
        Assert.Null(actions[0].Upgradable);
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
