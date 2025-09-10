using OGameSim.Entities;
using PlanningPlayer;
using Xunit;
using System.Collections.Generic;

namespace Tests;

public sealed class MctsPlannerTests
{
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
        var planner = new MctsPlanner(iterations: 50, maxDepth: 5);
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
        var few = new MctsPlanner(iterations: 20, maxDepth: 5);
        var many = new MctsPlanner(iterations: 200, maxDepth: 5);

        const int horizon = 20;
        var planFew = few.Plan(player, horizon);
        var planMany = many.Plan(player, horizon);

        var scoreFew = ExecutePlan(player, planFew);
        var scoreMany = ExecutePlan(player, planMany);

        Assert.True(scoreMany >= scoreFew);
    }

    [Fact]
    public void Planner_handles_long_horizon()
    {
        var player = new Player();
        var planner = new MctsPlanner(iterations: 100, maxDepth: 30);
        const int horizon = 8000;
        var plan = planner.Plan(player, horizon);
        var score = ExecutePlan(player, plan);
        Assert.True(score > 4_000m);
    }
}

