using OGameSim.Entities;
using MemoizationPlayer;

var player = new Player();
var result = Planner.Search(player, 8000);
Console.WriteLine(result);
