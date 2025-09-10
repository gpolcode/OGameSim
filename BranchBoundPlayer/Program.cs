using OGameSim.Entities;
using BranchBoundPlayer;

var player = new Player();
var result = Planner.Search(player, 8000);
Console.WriteLine(result);
