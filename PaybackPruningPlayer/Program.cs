using OGameSim.Entities;
using PaybackPruningPlayer;

var player = new Player();
var result = Planner.Search(player, 10);
Console.WriteLine(result);
