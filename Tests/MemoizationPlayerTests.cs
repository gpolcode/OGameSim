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
}
