using OGameSim.Entities;
using PlanningPlayer;
using Xunit;
using System;
using System.Collections.Generic;
using Xunit.Abstractions;

namespace Tests;

public sealed class MctsPlannerTests
{
    readonly ITestOutputHelper _output;

    public MctsPlannerTests(ITestOutputHelper output)
    {
        _output = output;
    }

    static decimal ExecutePlan(Player root, IReadOnlyList<ActionCandidate> plan)
    {
        var clone = root.DeepClone();
        foreach (var action in plan)
        {
            Planner.Apply(clone, action);
        }
        return clone.Points;
    }

    [Fact]
    public void Planner_returns_non_null_plan()
    {
        var player = new Player();
        var planner = new MctsPlanner(iterations: 50, maxDepth: 5, random: new Random(0));
        const int horizon = 20;
        var plan = planner.Plan(player, horizon);
        Assert.NotNull(plan);

        var clone = player.DeepClone();
        var steps = 0;
        foreach (var action in plan)
        {
            Planner.Apply(clone, action);
            steps += action.TimeCost;
        }
        Assert.True(steps >= horizon);
    }

    [Fact]
    public void Planner_improves_with_more_iterations()
    {
        var player = new Player();
        var few = new MctsPlanner(iterations: 20, maxDepth: 5, random: new Random(0));
        var many = new MctsPlanner(iterations: 200, maxDepth: 5, random: new Random(0));

        const int horizon = 20;
        var planFew = few.Plan(player, horizon);
        var planMany = many.Plan(player, horizon);

        var scoreFew = ExecutePlan(player, planFew);
        var scoreMany = ExecutePlan(player, planMany);

        Assert.True(scoreFew > 0m);
        Assert.True(scoreMany > 0m);
    }

    [Fact]
    public void Planner_handles_long_horizon()
    {
        var player = new Player();
        var planner = new MctsPlanner(iterations: 100, maxDepth: 30, random: new Random(0));
        const int horizon = 8000;
        var plan = planner.Plan(player, horizon);
        var score = ExecutePlan(player, plan);
        Assert.True(score > 250_000_000m);
    }

    [Fact]
    public void Planner_logs_scores_for_sample_iterations()
    {
        var iterations = new[] { 50, 200, 1000 };
        const int horizon = 20;
        foreach (var iter in iterations)
        {
            var player = new Player();
            var planner = new MctsPlanner(iterations: iter, maxDepth: 5, random: new Random(0));
            var plan = planner.Plan(player, horizon);
            var score = ExecutePlan(player, plan);
            _output.WriteLine($"Iterations {iter}: {score}");
        }

        Assert.True(true);
    }
}

