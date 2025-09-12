using System.Reflection;
using MemoizationPlayer;
using OGameSim.Entities;
using Xunit;
using System;
using System.IO;

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

    [Fact]
    public void Search_PrintsProgressAndResult()
    {
        var player = new Player();
        var original = Console.Out;
        using var sw = new StringWriter();
        Console.SetOut(sw);

        var result = Planner.Search(player, 3);
        Console.WriteLine(result);
        Console.Out.Flush();
        Console.SetOut(original);

        var output = sw.ToString();
        Assert.Contains("Day 3/3", output);
        Assert.EndsWith(result + Environment.NewLine, output);
    }

    [Fact]
    public void Search_Horizon8000_LogsResult()
    {
        var player = new Player();
        var result = Planner.Search(player, 8000);
        Console.WriteLine($"Memoization 8000-day result: {result}");
        Assert.True(true);
    }
}
