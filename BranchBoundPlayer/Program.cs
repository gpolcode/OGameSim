using OGameSim.Entities;
using BranchBoundPlayer;

var player = new Player();
var result = Planner.Search(player, 10);
Console.WriteLine(result);
