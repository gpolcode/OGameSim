using OGameSim.Entities;
using PaybackPruningPlayer;

var player = new Player();
var result = Planner.Search(player, 8000);
Console.WriteLine(result);
