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
}
