using OGameSim.Entities;
using PlanningPlayer;

// Demonstration entry point that plans with Monte Carlo tree search.
var player = new Player();

// Configure planner parameters.
var planner = new MctsPlanner(iterations: 100, maxDepth: 30);
const int horizon = 8000; // ~308M points with these settings

// Find plan and apply it to the player.
var plan = planner.Plan(player, horizon);
foreach (var action in plan)
{
    Planner.Apply(player, action);
}

Console.WriteLine($"Points after {horizon} steps: {player.Points}");
