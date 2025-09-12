using BranchBoundPlayer;
using OGameSim.Entities;
using Xunit;
using System;
using System.IO;

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

    [Fact]
    public void EnumerateActions_IncludesAstrophysics()
    {
        var player = new Player();
        var actions = Planner.EnumerateActions(player);
        Assert.Contains(actions, a => a.Upgradable == player.Astrophysics);
    }

    [Fact]
    public void Search_LogsBestScoreAndResult()
    {
        var player = new Player();
        var original = Console.Out;
        using var sw = new StringWriter();
        Console.SetOut(sw);

        var result = Planner.Search(player, 3);
        Console.WriteLine(result);
        Console.SetOut(original);

        var output = sw.ToString();
        Assert.Contains("New best score", output);
        Assert.EndsWith(result + Environment.NewLine, output);
    }

    [Fact(Skip = "Long-running demo")]
    public void Search_Horizon8000_LogsResult()
    {
        var player = new Player();
        var result = Planner.Search(player, 8000);
        Console.WriteLine($"BranchBound 8000-day result: {result}");
        Assert.True(true);
    }
}
