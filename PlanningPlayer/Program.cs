using OGameSim.Entities;
using PlanningPlayer;

// Demonstration entry point that plans with Monte Carlo tree search.
var player = new Player();

// Configure planner parameters.
var planner = new MctsPlanner(iterations: 1000, maxDepth: 5);
const int horizon = 20;

// Find plan and apply it to the player.
var plan = planner.Plan(player, horizon);
foreach (var action in plan)
{
    Planner.Apply(player, action);
}

Console.WriteLine($"Points after plan: {player.Points}");
