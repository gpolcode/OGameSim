using OGameSim.Entities;
using PlanningPlayer;
using Xunit;
using System.Collections.Generic;

namespace Tests;

public sealed class MctsPlannerTests
{
    static decimal ExecutePlan(Player root, IReadOnlyList<ActionCandidate> plan, int horizon)
    {
        var clone = root.DeepClone();
        foreach (var action in plan)
        {
            Planner.Apply(clone, action);
        }
        while (clone.Day < horizon)
        {
            var next = Planner.EnumerateActions(clone)[0];
            Planner.Apply(clone, next);
        }
        return clone.Points;
    }

    [Fact]
    public void Planner_returns_non_null_plan()
    {
        var player = new Player();
        var planner = new MctsPlanner(iterations: 50, maxDepth: 5);
        var plan = planner.Plan(player, horizon:20);
        Assert.NotNull(plan);
        Assert.NotEmpty(plan);
    }

    [Fact]
    public void Planner_improves_with_more_iterations()
    {
        var player = new Player();
        var few = new MctsPlanner(iterations: 20, maxDepth: 5);
        var many = new MctsPlanner(iterations: 200, maxDepth: 5);

        var planFew = few.Plan(player, horizon:20);
        var planMany = many.Plan(player, horizon:20);

        var scoreFew = ExecutePlan(player, planFew, 20);
        var scoreMany = ExecutePlan(player, planMany, 20);

        Assert.True(scoreMany >= scoreFew);
    }
}

