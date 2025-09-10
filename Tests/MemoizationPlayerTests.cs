using System.Reflection;
using MemoizationPlayer;
using OGameSim.Entities;
using Xunit;

public class MemoizationPlayerTests
{
    [Fact]
    public void BuildKey_IgnoresPoints()
    {
        var player = new Player();
        var key1 = Planner.BuildKey(player, 0);
        var prop = typeof(Player).GetProperty("Points", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!;
        prop.SetValue(player, 123m);
        var key2 = Planner.BuildKey(player, 0);
        Assert.Equal(key1, key2);
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
